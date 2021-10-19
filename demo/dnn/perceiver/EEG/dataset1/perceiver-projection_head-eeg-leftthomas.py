#! /bin/env/python

import numpy as np
import joblib
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from assess_eeg import load_dnn_data, assess_eeg
from perceiver_pytorch import Perceiver


class perceiver_projection_head(Perceiver):
    '''Perceiver with projection head.'''
    def __init__(self,  \
        perc_latent_array_dim = 200,\
        perc_num_latent_dim = 100,\
        perc_latent_heads = 8,\
        perc_depth = 6, \
        perc_weight_tie_layers = False,\
        perc_out_dim = 512,\
        proj_head_inp_dim = None, \
        proj_head_intermediate_dim = 512,\
        proj_head_out_dim = 200,\
        
        perc_cross_heads=1,\
        perc_cross_dim_head=64,\
        perc_latent_dim_head=64,\
        perc_attn_dropout=0,\
        perc_ff_dropout=0,\
        perc_self_per_cross_attn=2,\
        perc_num_freq_bands=6,\
        ): 
        

        
        self.perc_latent_array_dim = perc_latent_array_dim 
        self.perc_num_latent_dim = perc_num_latent_dim 
        self.perc_latent_heads = perc_latent_heads 
        self.perc_depth = perc_depth
        self.perc_weight_tie_layers = perc_weight_tie_layers
        self.perc_out_dim = perc_out_dim 
        self.proj_head_inp_dim = proj_head_inp_dim
        self.proj_head_intermediate_dim = proj_head_intermediate_dim  
        self.proj_head_out_dim = proj_head_out_dim 

        self.perc_cross_heads = perc_cross_heads
        self.perc_latent_dim_head = perc_latent_dim_head
        self.perc_latent_dim_head = perc_latent_dim_head
        self.perc_attn_dropout = perc_attn_dropout
        self.perc_ff_dropout = perc_ff_dropout
        self.perc_self_per_cross_attn = perc_self_per_cross_attn
        self.perc_num_freq_bands = perc_num_freq_bands

        if self.proj_head_inp_dim == None:
            self.proj_head_inp_dim = self.perc_out_dim 
        super(Perceiver, self).__init__()

        self.encoder = Perceiver(
            input_channels = 1,          # number of channels for each token of the input
            input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = perc_num_freq_bands,  # number of freq bands, with original value (2 * K + 1)
            max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = perc_depth,                   # depth of net
            num_latents = perc_num_latent_dim,           # number of latents, or induced set points, or centroids. 
            latent_dim = perc_latent_array_dim,            # latent dimension
            cross_heads = perc_cross_heads,             # number of heads for cross attention. paper said 1
            latent_heads = perc_latent_heads,            # number of heads for latent self attention, 8
            cross_dim_head = perc_cross_dim_head,
            latent_dim_head = perc_latent_dim_head,
            num_classes = perc_out_dim,          # output number of classesi = dimensionality of mvica output with 200 PCs
            attn_dropout = perc_attn_dropout,
            ff_dropout = perc_ff_dropout,
            weight_tie_layers = perc_weight_tie_layers, # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, 
            self_per_cross_attn = perc_self_per_cross_attn      # number of self attention blocks per cross attention
            )

        self.projection_head = nn.Sequential(nn.Linear(self.proj_head_inp_dim, proj_head_intermediate_dim, bias=False),\
            nn.BatchNorm1d(proj_head_intermediate_dim), nn.ReLU(inplace=True),\
            nn.Linear(proj_head_intermediate_dim, proj_head_out_dim, bias=True))

    def forward(self, inp):
        encoder_output = self.encoder(inp)
        projection_head_output = self.projection_head(encoder_output)
        return encoder_output, projection_head_output


class eeg_dataset_test(torch.utils.data.Dataset):
    '''EEG dataset for testing DNNs. 
    Getitem returns EEG 3d matrix of EEG responses of 
    all subjects for one (same) image presentation.'''
    def __init__(self, eeg_dataset, net, transform=None):
        '''
        Inputs:
        eeg_dataset - numpy array of eeg dataset of 
                      shape (subj, ims, chans, times)
        transformer - transformer to apply to the 
                      eeg dataset. If None, converts 
                      EEG dataset to tensor.
                      Default = None.
        net - str, type of net to use. resnet or perceiver.
            If resnet, getitem returns batches of shape
            (ims, 1, eeg_chan, time), where 1 is n channels

            If perceiver, getitem returns batches of shape
            (ims, eeg_chan, time, 1), where 1 is n channels
        '''
        self.net = net
        if transform == None:
            self.eeg = torch.tensor(eeg_dataset)
        else:
            self.eeg = transform(eeg_dataset)

    def __len__(self):
        '''Return number of subjects'''
        return self.eeg.shape[0]

    def __len_subj__(self):
        '''Return number of subjects.'''
        return self.eeg.shape[0]

    def __getitem__(self, subj):
        '''
        Inputs:
            subj - index along subject dimension.
        Ouputs: 
            out - 4d tensor of allimages for subject with index subj
        '''
        if self.net == 'resnet':
            out = self.eeg[subj,:,:,:].unsqueeze(1).type(torch.float32)
        elif self.net == 'perceiver':
            out = self.eeg[subj,:,:,:].unsqueeze(-1).type(torch.float32)
        return out


class eeg_dataset_train(torch.utils.data.Dataset):
    '''EEG dataset for tarining DNNs with contrasttive loss.
    Returns EEG response of 2 randomly picked 
    non-overlapping subjects.'''
    def __init__(self, eeg_dataset, net, transform = None, debug=False):
        '''
        Inputs:
            eeg_dataset - numpy array of eeg dataset of 
            shape (subj, ims, chans, times)
            transformer - transformer to apply to the 
                          eeg dataset. If None, converts 
                          eeg_dataset to tensor
            net - str, type of net to use. resnet or perceiver.
                If resnet, getitem returns batches of shape
                (idx, 1, eeg_chan, time), where 1 is n channels
                
                If perceiver, getitem returns batches of shape
                (idx, ch, time, eeg_chan, 1), where 1 is n channels
        Ouputs: 
            out1,out2 - 4d tensors of shape 
                resnet: (ims, chans, eeg_chans, times)
                perceiver: (ims, eeg_chans, times, chans)
        '''
        if transform == None:
            self.eeg = torch.tensor(eeg_dataset)
        else:
            self.eeg = self.transformer(eeg_dataset)
        self.debug = debug
        self.net = net

    def __len__(self):
        '''Return number of images'''
        return self.eeg.shape[1]

    def __getitem__(self, idx):
        '''
        idx - index along images
        '''
        subj_idx = np.random.permutation(np.linspace(0, self.eeg.shape[0],\
            self.eeg.shape[0], endpoint=False, dtype=int))

        batch1 = self.eeg[subj_idx[0],idx,:,:].type(torch.float32) # (idx, eeg_ch,time)
        batch2 = self.eeg[subj_idx[1],idx,:,:].type(torch.float32)
        if self.net == 'resnet':
            batch1 = batch1.unsqueeze(0)
            batch2 = batch2.unsqueeze(0)
        elif self.net == 'perceiver':
            batch1 = batch1.unsqueeze(-1)
            batch2 = batch2.unsqueeze(-1)
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
    elif normalize=="zscore":
        out1 = (out1 - torch.mean(out1,1).unsqueeze(1))/torch.std(out1, 1).unsqueeze(1)
        out2 = (out2 - torch.mean(out2,1).unsqueeze(1))/torch.std(out2, 1).unsqueeze(1)
    minval=1e-7
    concat_out =torch.cat([out1, out2], dim=0)
    sim_matrix = torch.exp(torch.mm(concat_out, concat_out.t().contiguous()).clamp(min=minval)/temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


    
def project_eeg(model, test_dataloader, layer="proj_head", split_size=None):
    '''
    Project EEG into new space independently for every subject using 
    trained model.
    Inputs:
        model - trained model (e.g. resnet50) Note, input dimensions required 
                by perceiver are different!
        test_dataloader - test dataloader for eeg_dataset_test class instance.
        layer - str, encoder or proj_head. Outputs of which layer to treat as
                projected EEG. Default = "proj_head".
        split_size - int, number of images per one call of the model. Helps
                     to reduce memory consumption and avoid cuda out of memory
                     error while projecting train set. If None, no separation 
                     into snippets is done. Default==None.
    Ouputs:
        projected_eeg - 3d numpy array of shape (subj, ims, features) 
                        of eeg projected into new (shared) space.
    '''
    model.eval()
    projected_eeg = []
    if split_size == None:
        for subj_data in test_dataloader:
            if torch.cuda.is_available():
                subj_data=subj_data.cuda()    
            feature, out = model(subj_data)
            if layer == "encoder":
                projected_eeg.append(feature.cpu().detach().numpy())
            elif layer == "proj_head":
                projected_eeg.append(out.cpu().detach().numpy())
        projected_eeg = np.stack(projected_eeg, axis=0)
    elif not split_size == None:
        for subj_data in test_dataloader:
            proj_eeg_tmp = []
            subj_data_list = torch.split(subj_data,  split_size, dim=0)
            for snippet in subj_data_list:
                if torch.cuda.is_available():
                    snippet=snippet.cuda()
                feature, out = model(snippet)
                if layer == "encoder":
                    proj_eeg_tmp.append(feature.cpu().detach().numpy())
                elif layer == "proj_head":
                    proj_eeg_tmp.append(out.cpu().detach().numpy())
            proj_eeg_tmp = np.concatenate(proj_eeg_tmp, axis=0)
            projected_eeg.append(proj_eeg_tmp)
        projected_eeg = np.stack(projected_eeg, axis=0)
    model.train()
    return projected_eeg


if __name__=='__main__':
    import argparse
    import time
    import warnings
    import copy
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir',type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/leftthomas/draft/',
    help='/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/EEG/leftthomas/draft/')
    parser.add_argument('-n_workers', type=int, default=0, help='default=0')
    parser.add_argument('-batch_size', type=int, default=16, help='Default=16')
    parser.add_argument('-gpu', action='store_true', default=False, help='Falg, whether to '
    'use GPU. Default = False.')
    parser.add_argument('-temperature',type=float, default=0.5, help='Temperature parameter for '
    'contrastive Loss. Default = 0.5')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate. Default=0.001')
    parser.add_argument('-n_epochs',type=int, default=10, help='How many times to pass '
    'through the dataset. Default=10')
    parser.add_argument('-bpl','--batches_per_loss',type=int, default=20, help='Save loss every '
    'bacth_per_loss mini-batches. Default=20.')
    parser.add_argument('-epta','--epochs_per_test_accuracy',type=int, default=1, help='Save test '
    'set accuracy every epochs_per_test_accuracy  epochs. Default == 1')
    parser.add_argument('-eeg_dir', type=str, default=\
    "/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/50hz/time_window13-40/",\
    help='Directory of EEG dataset to use. Default = /scratch/akitaitsev/intersubject_generalization/'
    'linear/dataset2/dataset_matrices/50hz/time_window13-40/') 
    parser.add_argument('-perc_latent_array_dim', type=int, default=200, help='Dimensions of latent query array '
    'of encoder. Default=200.')
    parser.add_argument('-perc_num_latent_dim', type=int, default=100, help='Number of latents, or induced '
    'set points, or centroids. Default=100.')
    parser.add_argument('-perc_latent_heads', type=int, default=8, help='Number of latent cross-attention heads. '
    'Default=8.')
    parser.add_argument('-perc_depth', type=int, default=6, help='Depth of net. Default=6.')
    parser.add_argument('-perc_share_weights', type=int, default=0, help='int, whether to share weights in perceiver.'
    'Default = 0.')
    parser.add_argument('-out_dim_ENC','--out_dim_encoder', type=int, default=512, help='Output dimensions of encoder. If no '
    'projection head, final output dimensions. Default=200.')
    parser.add_argument('-proj_head_intermediate_dim', type=int, default=512, help='Number of dims in intermediate layer of '
    'projection head.Default=512.')
    parser.add_argument('-out_dim_PH', '--out_dim_proj_head', type=int, default=200, help='Output dimensions of encoder. If no '
    'projection head, final output dimensions. Default=200.')
    parser.add_argument('-perc_cross_heads', type=int, default=1, help='Number of heads for cross-attention. Default=1.')
    parser.add_argument('-perc_cross_dim_head', type=int, default=64)
    parser.add_argument('-perc_latent_dim_head', type=int, default=64)
    parser.add_argument('-perc_self_per_cross_attn', type=int, default=2, help='Number of self-attention blocks per '
    'cross-attention. Default=2.')
    parser.add_argument('-perc_attn_dropout', type=int, default=0.) 
    parser.add_argument('-perc_ff_dropout', type=int, default=0.) 
    parser.add_argument('-perc_num_freq_bands', type=int, default=6, help='Number of freq bands, with original value (2 * K + 1).') 
    parser.add_argument('-clip_grad_norm', type=int, default=None, help='Int, value for gradient clipping by norm. Default=None, '
    'no clipping.')
    parser.add_argument('-pick_best_net_state',  action='store_true', default=False, help='Flag, whether to pick up the model with best '
    'generic decoding accuracy on encoder projection head layer over epta epochs to project the data. If false, uses model at '
    'the last epoch to project dadta. Default=False.')
    args=parser.parse_args()

    n_workers=args.n_workers
    batch_size=args.batch_size
    gpu=args.gpu
    learning_rate=args.lr
    out_dir=Path(args.out_dir)
    n_epochs = args.n_epochs
    bpl = args.batches_per_loss
    epta = args.epochs_per_test_accuracy
    temperature = args.temperature
    out_dim_ENC = args.out_dim_encoder
    out_dim_PH = args.out_dim_proj_head

    # EEG datasets
    datasets_dir = Path(args.eeg_dir)
    data_train = joblib.load(datasets_dir.joinpath('dataset_train.pkl'))
    data_test = joblib.load(datasets_dir.joinpath('dataset_test.pkl'))
   
    dataset_train = eeg_dataset_train(data_train, net='perceiver')
    dataset_test = eeg_dataset_test(data_test, net='perceiver')
    dataset_train_for_assessment = eeg_dataset_test(data_train, net='perceiver')

    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,\
                                                shuffle=True, num_workers=n_workers,\
                                                drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=None, \
                                                shuffle = False, num_workers=n_workers,\
                                                drop_last=False)
    train_dataloader_for_assessment = torch.utils.data.DataLoader(dataset_train_for_assessment,\
                                                batch_size=None, shuffle = False, num_workers=n_workers,\
                                                drop_last=False)

    # DNN data for regression
    dnn_dir='/scratch/akitaitsev/encoding_Ale/dataset1/dnn_activations/'
    X_train, X_val, X_test = load_dnn_data('CORnet-S', 1000, dnn_dir)

    # logging
    writer = SummaryWriter(out_dir.joinpath('runs'))    

    # define the model
    model = perceiver_projection_head(perc_latent_array_dim = args.perc_latent_array_dim,\
                                        perc_num_latent_dim = args.perc_num_latent_dim,\
                                        perc_latent_heads = args.perc_latent_heads,\
                                        perc_depth = args.perc_depth,\
                                        perc_weight_tie_layers = bool(args.perc_share_weights),\
                                        perc_out_dim = out_dim_ENC,\
                                        proj_head_intermediate_dim = args.proj_head_intermediate_dim,\
                                        proj_head_out_dim = out_dim_PH,\
                                        
                                        perc_cross_heads = args.perc_cross_heads,\
                                        perc_cross_dim_head = args.perc_cross_dim_head,\
                                        perc_latent_dim_head = args.perc_latent_dim_head,\
                                        perc_attn_dropout = args.perc_attn_dropout,\
                                        perc_ff_dropout = args.perc_ff_dropout,\
                                        perc_self_per_cross_attn = args.perc_self_per_cross_attn,\
                                        perc_num_freq_bands = args.perc_num_freq_bands,\
                                        )

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    
    # Loss and  accuracy init
    losses = defaultdict()
    accuracies = defaultdict()
    accuracies["encoder"] = defaultdict()
    accuracies["encoder"]["average"] = []
    accuracies["encoder"]["subjectwise"] = defaultdict()
    accuracies["encoder"]["subjectwise"]["mean"] = []
    accuracies["encoder"]["subjectwise"]["SD"] = []
    accuracies["projection_head"] = defaultdict()
    accuracies["projection_head"]["average"] = [] 
    accuracies["projection_head"]["subjectwise"] = defaultdict()
    accuracies["projection_head"]["subjectwise"]["mean"] = []
    accuracies["projection_head"]["subjectwise"]["SD"] = []
    cntr_epta=0 
    net_states=[]

    # Loop through EEG dataset in batches
    for epoch in range(n_epochs):
        model.train()
        cntr=0
        tic = time.time()
        losses["epoch"+str(epoch)]=[]

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
            
            # gradient clipping
            if args.clip_grad_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm, \
                norm_type=2.0)

            optimizer.step()
           
            # save loss every bpl mini-batches
            if cntr % bpl == 0 and cntr != 0:
                writer.add_scalar('training_loss', sum(losses["epoch"+str(epoch)][-bpl:])/bpl, cntr+\
                len(train_dataloader)*epoch) 
                toc=time.time() - tic
                tic=time.time()
                print('Epoch {:d}, average loss over bacthes {:d} - {:d}: {:.3f}. Time, s: {:.1f}'.\
                format(epoch, int(cntr-bpl), cntr, sum(losses["epoch"+str(epoch)][-bpl:])/bpl, toc))
            cntr+=1

        # save test accuracy every epta epcohs
        if cntr_epta % epta == 0:

            tic = time.time()

            # Project train and test set EEG into new space

            # treat encoder output as EEG
            eeg_train_projected_ENC = project_eeg(model, train_dataloader_for_assessment, layer="encoder", split_size=25) 
            eeg_test_projected_ENC = project_eeg(model, test_dataloader, layer = "encoder")  

            # treat projection head output as EEG
            eeg_train_projected_PH = project_eeg(model, train_dataloader_for_assessment, split_size=25) 
            eeg_test_projected_PH = project_eeg(model, test_dataloader)  
            
            av_ENC, sw_ENC = assess_eeg(X_train, X_test, eeg_train_projected_ENC, eeg_test_projected_ENC)
            av_PH, sw_PH = assess_eeg(X_train, X_test, eeg_train_projected_PH, eeg_test_projected_PH)

            accuracies["encoder"]["average"].append(av_ENC[0])
            accuracies["encoder"]["subjectwise"]["mean"].append(sw_ENC[0])
            accuracies["encoder"]["subjectwise"]["SD"].append(sw_ENC[1])
            accuracies["projection_head"]["average"].append(av_PH[0])
            accuracies["projection_head"]["subjectwise"]["mean"].append(sw_PH[0])
            accuracies["projection_head"]["subjectwise"]["SD"].append(sw_PH[1])
            
            # Print info
            toc = time.time() - tic
            print('Network top1 generic decoding accuracy on encoder output at epoch {:d}:\n'
            'Average: {:.2f} % \n Subjectwise: {:.2f} % +- {:.2f} (SD)'.format(epoch, av_ENC[0], sw_ENC[0], sw_ENC[1]))
            print('Network top1 generic decoding accuracy on proj_head output at epoch {:d}:\n'
            'Average: {:.2f} % \n Subjectwise: {:.2f} % +- {:.2f} (SD)'.format(epoch, av_PH[0], sw_PH[0], sw_PH[1]))
            print('Elapse time: {:.2f} minutes.'.format(toc/60))   

            # logging 
            writer.add_scalar('accuracy_encoder_av', av_ENC[0],\
                    len(train_dataloader)*cntr_epta) 
            writer.add_scalar('accuracy_encoder_sw', sw_ENC[0],\
                    len(train_dataloader)*cntr_epta) 
            writer.add_scalar('accuracy_proj_head_av', av_PH[0],\
                    len(train_dataloader)*cntr_epta) 
            writer.add_scalar('accuracy_proj_head_sw', sw_PH[0],\
                    len(train_dataloader)*cntr_epta) 
            
            if args.pick_best_net_state:
                net_states.append(copy.deepcopy(model.state_dict()))

        cntr_epta += 1
    writer.close()

    # select net state which yieled best accuracy on encoder average 
    if args.pick_best_net_state:
        best_net_state = net_states[ np.argmax(accuracies["encoder"]["average"]) ]
        model.load_state_dict(best_net_state)

    # Project EEG into new space using trained model
    projected_eeg = defaultdict()
    projected_eeg["train"] = defaultdict()
    projected_eeg["test"] = defaultdict() 
    projected_eeg["train"]["encoder"] = project_eeg(model, train_dataloader_for_assessment, layer="encoder", split_size=25) 
    projected_eeg["train"]["projection_head"] = project_eeg(model, train_dataloader_for_assessment, layer="encoder", split_size=25) 
    projected_eeg["test"]["encoder"] = project_eeg(model, test_dataloader, layer="proj_head") 
    projected_eeg["test"]["projection_head"] = project_eeg(model, test_dataloader, layer="proj_head") 
    
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    # save projected EEG 
    joblib.dump(projected_eeg, out_dir.joinpath('projected_eeg.pkl'))

    # save trained model
    torch.save(model.state_dict(), out_dir.joinpath('trained_model.pt'))

    # save loss profile
    joblib.dump(losses, out_dir.joinpath('losses.pkl')) 
    
    # save test accuracy profile
    joblib.dump(accuracies, out_dir.joinpath('test_accuracies.pkl'))
