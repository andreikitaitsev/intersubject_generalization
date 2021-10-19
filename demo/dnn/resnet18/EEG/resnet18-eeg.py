#! /bin/env/python

import numpy as np
import joblib
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import sklearn
import sklearn.discriminant_analysis
import sklearn.neighbors
import sklearn.svm
import copy
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
from pathlib import Path
from assess_eeg import assess_eeg

from resnet import resnet18 

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
 
class dataset_contrastive_loss(torch.utils.data.Dataset):
    '''Dataset for contrastive loss learning. 
    getitem returns 2 augmented versions of the same image'''
    def __init__(self, data, transformer=None):
        '''
        data - 4d tensor of images (images, pix, pix, chans)
        transformer - random transformer object which 
                      will be called on every geiitem call'''
        self.data = data
        if transformer == None:
            self.transformer = transforms.ToTensor()
        else:
            self.transformer = transformer
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        augm1 = self.transformer(self.data[idx,:,:,:])
        augm2 = self.transformer(self.data[idx,:,:,:])
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

class resnet18_projection_head(nn.Module):
    ''' 
    Modified resnet 50 with user defined number of input channels and  output channels 
    for each layer of the  net.
    
    Input size shall be (images, channels, pix, pix).
    '''
    def __init__(
        self,
        inp_chs_conv1=1,
        out_chs_conv1=64,
        out_chs1=64,
        out_chs2=128,
        out_chs3=256,
        out_chs4=512,
        proj_head_inp_dim=2048, 
        proj_head_intermediate_dim=512, 
        feature_dim=128):
        super(resnet18_projection_head, self).__init__()
        self.register_buffer('proj_head_inp_dim', torch.tensor(proj_head_inp_dim))
        # encoder
        self.f = resnet18(pretrained=False, inp_chs_conv1 = inp_chs_conv1, out_chs_conv1=out_chs_conv1, out_chs1=out_chs1,\
            out_chs2 = out_chs2, out_chs3 = out_chs3, out_chs4=out_chs4)
        # projection head
        self.g = nn.Sequential(nn.Linear(proj_head_inp_dim, proj_head_intermediate_dim, bias=False),\
                nn.BatchNorm1d(proj_head_intermediate_dim), nn.ReLU(inplace=True), \
                nn.Linear(proj_head_intermediate_dim, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        if self.proj_head_inp_dim != feature.shape[1]:
            feature = torch.squeeze(torch.nn.functional.interpolate(feature.unsqueeze(0), \
            int(self.proj_head_inp_dim)),0)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


def test_net_projection_head(model, clf,  train_dataloader_no_transform, test_dataloader):
    '''Test classification accuracy with any calssifier fit on output of both encoder and 
    projection head. Net shall output 2 feature sets: first - encoder output, second - 
    projection head output.
    Inputs:
        model - DNN model
        classifier - sklearn classifer object or list of classifer objects
        train_dataloader_no_transform - train dataloader without any data augmentation
        tets_dataloader
        classify_encoder_output -bool, whether to do classification on encoder 
            output. If False, classify projection head outputs. Default=True.
    Outputs:
        enc_acc - average classification accuracy on 1 epoch of encoder features on test set
        proj_head_acc - average classification accuracy on 1 epoch of projection head outputs on test set
    ''' 

    model.eval()
    train_features = []
    train_out = []
    train_targets = []
    test_features = []
    test_out = []
    test_targets = []

    if torch.cuda.is_available():
        device="cuda"
    with torch.no_grad():
        # create out and target array for the train dataset
        for data, target in train_dataloader_no_transform:
            feature, out = model(data.to(device)) # (ims, output_dim)
            train_features.append(feature.cpu().detach().numpy())
            train_out.append(out.cpu().detach().numpy())
            train_targets.append(target)
        train_features = np.concatenate(train_features, axis=0)
        train_out = np.concatenate(train_out, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        
        # create out and target array for the train dataset
        for data, target in test_dataloader:
            feature, out = model(data.to(device)) # (ims, output_dim)
            test_features.append(feature.cpu().detach().numpy())
            test_out.append(out.cpu().detach().numpy())
            test_targets.append(target)
        test_features = np.concatenate(test_features, axis=0)
        test_out = np.concatenate(test_out, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)
    
    clf_enc = copy.deepcopy(clf)
    clf_proj_head = copy.deepcopy(clf)
    clf_enc.fit(train_features, train_targets)
    clf_proj_head.fit(train_out, train_targets)

    # predict test targets from test output
    pred_targets_enc = clf_enc.predict(test_features)
    pred_targets_proj_head = clf_proj_head.predict(test_out)
    # average accuracy 
    enc_acc = sum(pred_targets_enc == test_targets)*100/len(test_targets)
    proj_head_acc = sum(pred_targets_proj_head == test_targets)*100/len(test_targets)
    return enc_acc, proj_head_acc

    
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
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir',type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/dnn/resnet18/EEG/dataset1/draft/',
    help='/scratch/akitaitsev/intersubject_generalization/dnn/resnet/EEG/dataset1/draft/')
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
    parser.add_argument('-inp_chs_conv1', type=int, default=1, help='Num input channels in resnet18. '
    'Default=1.')
    parser.add_argument('-out_chs_conv1', type=int, default=64, help='Num output channels in resnet18. '
    'Default=64.')
    parser.add_argument('-out_chs1', type=int, default=64, help='Num output channels in 1st resent18 layer. Default=64')
    parser.add_argument('-out_chs2', type=int, default=128, help='Num output channels in 2nd resent18 layer. Default=128')
    parser.add_argument('-out_chs3', type=int, default=256, help='Num output channels in 3rd resent18 layer. Default=256')
    parser.add_argument('-out_chs4', type=int, default=512, help='Num output channels in 4th resent18 layer. Default=512')
    parser.add_argument('-proj_head_inp_dim', type=int, default=2048, help='Input dim of projection head '
    'layer. Default=2048')
    parser.add_argument('-proj_head_intermediate_dim', type=int, default=512, help='Intermediate dim of '
    'projection head layer. Default=512.')
    parser.add_argument('-feature_dim', type=int, default=200, help='Num output features in projection '
    'head layer of the model. Defautl = 200.') 
    parser.add_argument('-pick_best_net_state',  action='store_true', default=False, help='Flag, whether to pick '
    'up the model with best generic decoding accuracy on encoder projection head layer over epta epochs to '
    'project the data. If false, uses model at the last epoch to project dadta. Default=False.')
    parser.add_argument('-eeg_dir',  type=str, default='/scratch/akitaitsev/intersubject_generalization/linear/'
    '/dataset1/dataset_matrices/50hz/time_window13-40/', help='EEG dataset dir. Default= '
    '/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/50hz/time_window13-40/')
    args=parser.parse_args()

    featuredim=args.feature_dim
    n_workers=args.n_workers
    batch_size=args.batch_size
    gpu=args.gpu
    learning_rate=args.lr
    out_dir=Path(args.out_dir)
    n_epochs = args.n_epochs
    bpl = args.batches_per_loss
    epta = args.epochs_per_test_accuracy
    temperature = args.temperature

    # EEG datasets
    datasets_dir = Path(args.eeg_dir)
    data_train = joblib.load(datasets_dir.joinpath('dataset_train.pkl'))
    data_test = joblib.load(datasets_dir.joinpath('dataset_test.pkl'))
   
    dataset_train = eeg_dataset_train(data_train, net='resnet')
    dataset_test = eeg_dataset_test(data_test, net='resnet')
    dataset_train_for_assessment = eeg_dataset_test(data_train, net='resnet')

    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,\
                                                shuffle=True, num_workers=n_workers,\
                                                drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=None, \
                                                shuffle = False, num_workers=n_workers,\
                                                drop_last=False)
    train_dataloader_for_assessment = torch.utils.data.DataLoader(dataset_train_for_assessment,\
                                                batch_size=None, shuffle = False, num_workers=n_workers,\
                                                drop_last=False)

    # logging
    writer = SummaryWriter(out_dir.joinpath('runs'))    

    # define the model
    model = resnet18_projection_head(inp_chs_conv1=args.inp_chs_conv1, out_chs_conv1=args.out_chs_conv1,\
        out_chs1=args.out_chs1, out_chs2= args.out_chs2, out_chs3 = args.out_chs3, out_chs4=args.out_chs4,\
        proj_head_inp_dim=args.proj_head_inp_dim, proj_head_intermediate_dim=\
        args.proj_head_intermediate_dim, feature_dim=args.feature_dim)

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
        accuracies["epoch"+str(epoch)]=[]
        batch_counter=0
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
            eeg_train_projected_ENC = project_eeg(model, train_dataloader_for_assessment, layer="encoder", split_size=100) 
            eeg_test_projected_ENC = project_eeg(model, test_dataloader, layer = "encoder")  

            # treat projection head output as EEG
            eeg_train_projected_PH = project_eeg(model, train_dataloader_for_assessment, split_size=100) 
            eeg_test_projected_PH = project_eeg(model, test_dataloader)  
            
            av_ENC, sw_ENC = assess_eeg(eeg_train_projected_ENC, eeg_test_projected_ENC)
            av_PH, sw_PH = assess_eeg(eeg_train_projected_PH, eeg_test_projected_PH)

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
    projected_eeg["train"]["encoder"] = project_eeg(model, train_dataloader_for_assessment, layer="encoder", split_size=100) 
    projected_eeg["train"]["projection_head"] = project_eeg(model, train_dataloader_for_assessment, layer="encoder", split_size=100) 
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
