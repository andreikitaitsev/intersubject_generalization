#! /bin/env/python

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.models.resnet import resnet50
from pathlib import Path
import joblib


class eeg_trainset(torch.utils.data.Dataset):
    '''EEG dataset class.'''
    def __init__(self, eeg_dataset, transform=None, debug=False):
        '''
        Inputs:
            eeg_dataset - numpy array of eeg dataset of 
            shape (subj, ims, chans, times)
            debug - print used indices
            transformer - transformer to apply to the dataset.
                          If none, converts it into tensor
        '''
        if transform == None:
            self.transformer = transforms.ToTensor()
        else:
            self.transformer = transform
        self.eeg = self.transformer(eeg_dataset)
        self.debug = debug

    def __len__(self):
        '''Return number of images'''
        return self.eeg.shape[1]

    def __getitem__(self, idx):
        '''
        idx - index along images
        '''
        subj_idx = np.random.permutation(np.linspace(0, self.eeg.shape[0],\
            self.eeg.shape[0], endpoint=False, dtype=int))
        batch1 = self.eeg[subj_idx[0],idx,:,:].type(torch.float32)
        batch2 = self.eeg[subj_idx[1],idx,:,:].type(torch.float32)
        if self.debug:
            print("Subject indices to be shuffled: "+' '.join(map(str, subj_idx.tolist())))
            print("Indexing for batch 1: [{:d}:{:d},:,:]".format(\
                subj_idx[0], idx))
            print("Indexing for batch 2: [{:d}:{:d},:,:]".format(\
                subj_idx[1], idx))
        return (batch1, batch2)

class ContrastiveLoss_zablo(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5, device="cpu"):
        '''Inputs:
            batch_size - int,
            temperature -float
            device - str, cpu or cuda
        '''
        super().__init__()
        self.batch_size = torch.tensor(batch_size)
        self.temperature = torch.tensor(temperature)
        self.negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        self.device=torch.device(device)
        self.device_name=device
        if "cuda" in device:
            self.batch_size = self.batch_size.cuda()
            self.temperature = self.temperature.cuda()
            self.negatives_mask = self.negatives_mask.cuda() 
    
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        if "cpu" in self.device_name:
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
        elif "cuda" in self.device_name:
            z_i = F.normalize(emb_i, dim=1)
            z_j = F.normalize(emb_j, dim=1)
            z_i=z_i.cuda()
            z_j=z_j.cuda()
            representations = torch.cat([z_i, z_j], dim=0)
            representations=representations.cuda()
            similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
            similarity_matrix = similarity_matrix.cuda()
            sim_ij = torch.diag(similarity_matrix, self.batch_size)
            sim_ji = torch.diag(similarity_matrix, -self.batch_size)
            sim_ij=sim_ij.cuda()
            sim_ji=sim_ji.cuda()
            positives = torch.cat([sim_ij, sim_ji], dim=0)
            positives=positives.cuda()
            nominator = torch.exp(positives / self.temperature)
            nominator=nominator.cuda()
            denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
            denominator = denominator.cuda()
            loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
            loss_partial = loss_partial.cuda()
            loss = torch.sum(loss_partial) / (2 * self.batch_size)
            loss=loss.cuda()
        return loss
 
class dataset_contrastive_loss(torch.utils.data.Dataset):
    '''Dataset for contrastive loss learning. 
    getitem returns 2 augmented versions of the same image'''
    def __init__(self, data, transformer=None):
        '''data - 4d tensor of images (images, channels, pix, pix)
        transformer - random transformer object which 
        will be called on every geiitem call'''
        self.data = data
        self.transformer = transformer
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        if self.transformer !=None:
            augm1 = self.transformer(self.data[idx,:,:,:])
            augm2 = self.transformer(self.data[idx,:,:,:])
        elif self.transformer == None:
            augm1 = self.data[idx,:,:,:]
            augm2 = self.data[idx,:,:,:]
        return (augm1, augm2)

def ContrastiveLoss_leftthomas(out1, out2, batch_size, temperature, normalize="normalize"):
    '''Inputs:
        out1, out2 - outputs of dnn input to which were batches of images yielded
                     from contrastive_loss_dataset
        batch_size - int
        temperature -int
        normalize - normalize or zscore
    '''
    if normalize=="normalize":
        out1=F.normalize(out1, dim=1) 
        out2=F.normalize(out2, dim=1)
    concat_out =torch.cat([out1, out2], dim=0)
    sim_matrix = torch.exp(torch.mm(concat_out, concat_out.t().contiguous()))/temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


if __name__=='__main__':
    import argparse
    import time
    import warnings
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir',type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/trial3/CIFAR/leftthomas/',
    help='default=/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/trial3/CIFAR/leftthomas/')
    parser.add_argument('-n_workers', type=int, default=0, help='default=0')
    parser.add_argument('-batch_size', type=int, default=16, help='Default=16')
    parser.add_argument('-gpu', action='store_true', default=False, help='Falg, whether to '
    'use GPU. Default = False.')
    parser.add_argument('-temperature',type=float, default=0.5, help='Temperature parameter for '
    'contrastive Loss. Default = 0.5')
    parser.add_argument('-lr', type=float, default=0.01, help='Learning rate. Default=0.01')
    parser.add_argument('-n_epochs',type=int, default=10, help='How many times to pass '
    'through the dataset. Default=10')
    parser.add_argument('-bpl','--batches_per_loss',type=int, default=20, help='Save loss every '
    'bacth_per_loss mini-batches. Default=20.')
    parser.add_argument('-bpta','--batches_per_test_accuracy',type=int, default=None, help='Save test '
    'set accuracy every bacthes_per_test_accuracy  mini-batches. Default==batches_per_loss.')
    parser.add_argument('-feature_dim', type=int, default=128, help='Dimensionality of the latent '
    'projection head. Default=128')
    args=parser.parse_args()

    featuredim = args.feature_dim 
    n_workers=args.n_workers
    batch_size=args.batch_size
    gpu=args.gpu
    learning_rate=args.lr
    out_dir=Path(args.out_dir)
    n_epochs = args.n_epochs
    bpl = args.batches_per_loss
    if args.batches_per_test_accuracy == None:
        args.batches_per_test_accuracy = args.batches_per_loss
    else:
        warnings.warn('Batches_per_test_accuracy parameters is set manually. By default '
        'it is equal to batches_per_loss.')
    bpta = args.batches_per_test_accuracy

    # logging
    writer = SummaryWriter(out_dir.joinpath('runs'))    
    
    # EEG datasets  
    datasets_dir = Path('/scratch/akitaitsev/intersubject_generalizeation/linear/',\
    'dataset_matrices/50hz/time_window13-40/')

    trainset = jo   dataset_train = joblib.load(datasets_dir.joinpath('dataset_train.pkl'))
    dataset_train = joblib.load(datasets_dir.joinpath('dataset_train.pkl'))
    dataset_test = joblib.load(datasets_dir.joinpath('dataset_test.pkl'))
    
    dataset_train = eeg_dataset(dataset_train)
    
    # Datasloaders
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
        shuffle=False, num_workers=n_workers, drop_last=True)

    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers,\
                                             drop_last=True)


    # define the model
    model = Model(featuredim)

    if gpu and n_workers >=1:
        warnings.warn('Using GPU and n_workers>=1 can cause some difficulties.')
    if gpu:
        device_name="cuda"
        device=torch.device("cuda:0")
        model.to(torch.device("cuda:0"))
        model=torch.nn.DataParallel(model) 
        print("Using "+str(torch.cuda.device_count()) +" GPUs.")
    elif not gpu: 
        device=torch.device("cpu")
        device_name="cpu"
        print("Using CPU.") 
    
    # define optmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Logging
    writer = SummaryWriter(out_dir.joinpath('runs'))
    
    # Loop through EEG dataset in batches
    losses = defaultdict()
    accuracies = defaultdict()
    for epoch in range(n_epochs):
        cntr=0
        cntr_test=0
        tic = time.time()
        losses["epoch"+str(epoch)]=[]
        accuracies["epoch"+str(epoch)]=[]
        for batch1, batch2 in train_dataloader:
            if args.gpu:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
            feature1, out1 = model.forward(batch1)
            feature2, out2 = model.forward(batch2)
           
            # compute loss
            loss = ContrastiveLoss_leftthomas(out1, out2, args.batch_size, args.temperature)
            losses["epoch"+str(epoch)].append(loss.cpu().detach().numpy())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            cntr+=1
            cntr_test+=1

            # save loss every bpl mini-batches
            if cntr % bpl == 0:
                writer.add_scalar('training_loss', sum(losses["epoch"+str(epoch)][-bpl:])/bpl, cntr+\
                len(train_dataloader)*epoch) 
                toc=time.time() - tic
                tic=time.time()
                print('Epoch {:d}, average loss over bacthes {:d} - {:d}: {:.3f}. Time, s: {:.1f}'.\
                format(epoch, int(cntr-bpl), cntr, sum(losses["epoch"+str(epoch)][-bpl:])/bpl, toc))

    writer.close()

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    # save trained model
    torch.save(model.state_dict(), out_dir.joinpath('trained_model.pt'))

    # save loss profile
    joblib.dump(losses, out_dir.joinpath('losses.pkl')) 
