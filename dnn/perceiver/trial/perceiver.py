#! /bin/env/python

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from perceiver_pytorch import Perceiver
from pathlib import Path
import joblib

class transformer(object):
    '''Transform eeg dataset when creating eeg_dataset class.'''
    def __init__(self, to_tensor=True):
        self.to_tensor = to_tensor
    def __call__(self, eeg_dataset):
        if self.to_tensor:
            self.eeg = torch.tensor(eeg_dataset)
        return self.eeg 
        

class eeg_dataset(torch.utils.data.Dataset):
    '''EEG dataset class.'''
    def __init__(self, eeg_dataset, transform=None, return_idx=False):
        '''
        Inputs:
            eeg_dataset - numpy array of eeg dataset of 
            shape (subj, ims, chans, times)
            transformer - transformer to apply to the 
                          eeg dataset. If None, reshapes
                          eeg_dataset into (-1, chans,times)
                          and converts it to tensor
        '''
        if transform == None:
            self.transformer = transformer()
        else:
            self.transformer = transform
        self.eeg = self.transformer(eeg_dataset)

    def __len__(self):
        '''Return number of images'''
        return self.eeg.shape[1]

    def __getitem__(self, idx):
        '''
        idx - index along images
        '''
        subj_idx = np.random.permutation(np.linspace(0,self.eeg.shape[0],\
            self.eeg.shape[0], endpoint=False, dtype=int))
        batch1 = self.eeg[subj_idx[0],idx,:,:]
        batch2 = self.eeg[subj_idx[1],idx,:,:]
        return (batch1, batch2)

class ContrastiveLoss_manual(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=True):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)

            denominator = torch.sum(
            one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
 


if __name__=='__main__':
    # for now manually define parameters
    batch_size=5
    temperature=0.5
    learning_rate=0.05
    
    
    
    
    datasets_dir = Path('/scratch/akitaitsev/intersubject_generalizeation/linear/',\
    'dataset_matrices/50hz/time_window13-40/')
    dataset_train = joblib.load(datasets_dir.joinpath('dataset_train.pkl'))
    dataset_test = joblib.load(datasets_dir.joinpath('dataset_test.pkl'))
    
    dataset_train = eeg_dataset(dataset_train)
    dataset_test = eeg_dataset(dataset_test)

    # define the model
    model = Perceiver(  
        input_channels = 1,          # number of channels for each token of the input
        input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
        num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
        max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
        depth = 6,                   # depth of net
        num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = 512,            # latent dimension
        cross_heads = 1,             # number of heads for cross attention. paper said 1
        latent_heads = 8,            # number of heads for latent self attention, 8
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 200,          # output number of classesi = dimensionality of mvica output with 200 PCs
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn = 2      # number of self attention blocks per cross attention
        )

    # define train and test dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
        shuffle=False, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
        shuffle=False, num_workers=1)
    
    # define loss function
    loss_fn = ContrastiveLoss(batch_size, temperature)
    
    # define optmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loop through EEG dataset in batches
    projections = []
    losses = []
    import ipdb; ipdb.set_trace() 
    for batch1, batch2 in train_dataloader:
       concat_batch=torch.cat((batch1, batch2), 0)
       concat_batch=concat_batch.unsqueeze(3)
       # otherwise model does not work somewhy
       concat_batch.to(torch.float32)
       projections = model(concat_batch)
       proj1 = projections[:batch_size] #images, features
       proj2 = projections[batch_size:]
       
       # compute loss
       loss = loss_fn.forward(proj1, proj2)
       losses.append(loss)

       optimizer.zero_grad()

       loss.backward()
       optimizer.step()


