#! /bin/env/python

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.svm import SVC
from perceiver_pytorch import Perceiver
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.models.resnet import resnet50
from pathlib import Path
import joblib


class eeg_dataset(torch.utils.data.Dataset):
    '''EEG dataset class.'''
    def __init__(self, eeg_dataset, transform=None, debug=False):
        '''
        Inputs:
            eeg_dataset - numpy array of eeg dataset of 
            shape (subj, ims, chans, times)
            transformer - transformer to apply to the 
                          eeg dataset. If None, reshapes
                          eeg_dataset into (-1, chans,times)
                          and converts it to tensor
            debug - print used indices
        '''
        if transform == None:
            self.transformer = transforms.ToTensor()
        else:
            self.transformer = transfortransforms.ToTensorm
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


def test_KNN_perceiver(model, test_dataloader):
    '''Test classification accuracy with KNN calssifier.''' 
    model.eval()
    total_top1, total_num, out_bank = 0.0, 0, []
    c = len(test_dataloader.dataset.classes)
    k = 200
    if torch.cuda.is_available():
        device="cuda"
    with torch.no_grad():
        # create feature bank
        for data, target in test_dataloader:
            data = data.permute(0,2,3,1) #from (ims, chans, height, weight)
            # to (ims, height, weight, chans)
            out = model(data.to(device))
            out_bank.append(out)
        out_bank = torch.cat(out_bank, dim=0).t().contiguous()
        out_labels = torch.tensor(test_dataloader.dataset.targets, device=out_bank.device)
        
        for data, target in test_dataloader: 
            data = data.to(device)
            data = data.permute(0,2,3,1)
            target = target.to(device)
            out = model(data)
            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(out, out_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(out_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
    return total_top1 / total_num * 100
    
def test_SVM_perceiver(model, train_dataloader_svm, test_dataloader):
    '''Test classification accuracy with SVM calssifier fit
    on train data representations.''' 
    model.eval()
    train_outs = []
    train_targets = []
    test_outs=[]
    test_targets=[]

    if torch.cuda.is_available():
        device="cuda"
    with torch.no_grad():
        # create out and target array for the train dataset
        for data, target in train_dataloader_svm:
            data = data.permute(0,2,3,1) #from (ims, chans, height, weight)
            # to (ims, height, weight, chans)
            out = model(data.to(device)) # (ims, output_dim)
            train_outs.append(out.cpu().detach().numpy())
            train_targets.append(target)
        train_outs = np.concatenate(train_outs, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        clf = SVC()
        clf.fit(train_outs, train_targets)

        # create out and target array for the train dataset
        for data, target in test_dataloader:
            data = data.permute(0,2,3,1) #from (ims, chans, height, weight)
            # to (ims, height, weight, chans)
            out = model(data.to(device)) # (ims, output_dim)
            test_outs.append(out.cpu().detach().numpy())
            test_targets.append(target)
        test_outs = np.concatenate(test_outs, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)

    # predict test targets from test output
    pred_targets = clf.predict(test_outs)
    
    # average accuracy 
    av_acc = sum(pred_targets == test_targets)*100/len(test_targets)
    return av_acc



if __name__=='__main__':
    import argparse
    import time
    import warnings
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir',type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/dnn/perceiver/trial5/CIFAR/leftthomas/draft/',
    help='default = /scratch/akitaitsev/intersubject_generalization/dnn/perceiver/trial5/CIFAR/leftthomas/draft/')
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
    parser.add_argument('-output_dim', type=int, default=200, help='Dimensionality of the output '
    'of the model. Default=200.')
    parser.add_argument('-clf','--classifier', type=str, default='KNN', help='Classifier to use on '
    'DNN outputs. KNN or SVM. Default=KNN.')
    args=parser.parse_args()

    clf = args.classifier
    n_workers=args.n_workers
    batch_size=args.batch_size
    gpu=args.gpu
    learning_rate=args.lr
    out_dir=Path(args.out_dir)
    n_epochs = args.n_epochs
    bpl = args.batches_per_loss
    epta = args.epochs_per_test_accuracy
    temperature = args.temperature

    # transforms
    weight = 30
    height = 30
    crop = torchvision.transforms.RandomCrop(weight, height)

    brightness = 0.1
    contrast = 0.1
    saturation = 0.1
    hue=0.1
    colorjitter = torchvision.transforms.ColorJitter(brightness, contrast, \
        saturation, hue)

    degrees=(-90,90)
    rotation = torchvision.transforms.RandomRotation(degrees)
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    # logging
    writer = SummaryWriter(out_dir.joinpath('runs'))    

    # random augmentation of training images
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True) 
    train_images = trainset.data # (ims, pix, pix, chans)
    trainset = dataset_contrastive_loss(train_images, train_transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=n_workers,\
                                              drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,\
                                                        transform=test_transform)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers,\
                                             drop_last=True)

    trainset_svm = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,\
                                            transform=test_transform)
    train_dataloader_svm = torch.utils.data.DataLoader(trainset_svm, batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers,\
                                             drop_last=True)

    # define the model
    
    # Potential hyperparameters
    output_dim=args.output_dim
    latent_array_dims=200
    num_latent_dims=100
    latent_heads=8

    model = Perceiver(  
        input_channels = 3,          # number of channels for each token of the input
        input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
        num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
        max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
        depth = 6,                   # depth of net
        num_latents = num_latent_dims,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = latent_array_dims,            # latent dimension
        cross_heads = 1,             # number of heads for cross attention. paper said 1
        latent_heads = latent_heads,            # number of heads for latent self attention, 8
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = output_dim,          # output number of classesi = dimensionality of mvica output with 200 PCs
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn = 2      # number of self attention blocks per cross attention
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
    
    # Logging
    writer = SummaryWriter(out_dir.joinpath('runs'))
    
    # Loop through dataset in batches
    losses = defaultdict()
    accuracies = defaultdict()
    cntr_epta=0
    for epoch in range(n_epochs):
        model.train()
        cntr=0
        tic = time.time()
        losses["epoch"+str(epoch)]=[]
        accuracies["epoch"+str(epoch)]=[]

        for batch1, batch2 in train_dataloader:
            batch1 = batch1.permute(0,2,3,1)
            batch2 = batch2.permute(0,2,3,1)
            if args.gpu:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
            out1 = model.forward(batch1)
            out2 = model.forward(batch2)
           
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
            if clf == 'KNN':
                acc = test_KNN_perceiver(model, test_dataloader)
                print('Network accuracy (KNN) at epoch {:d}: {:.2f} %'.\
                    format(epoch, acc))
                writer.add_scalar('KNN_test_accuracy', acc,  
                    len(train_dataloader)*cntr_epta) 
            elif clf == 'SVM':
                acc = test_SVM_perceiver(model, train_dataloader_svm, test_dataloader)
                print('Network accuracy (SVM) at epoch {:d}: {:.2f} %'.\
                    format(epoch, acc))
                writer.add_scalar('SVM_test_accuracy', acc,  
                        len(train_dataloader)*cntr_epta) 
        cntr_epta += 1
    writer.close()

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    # save trained model
    torch.save(model.state_dict(), out_dir.joinpath('trained_model.pt'))

    # save loss profile
    joblib.dump(losses, out_dir.joinpath('losses.pkl')) 
    
    # save test accuracy profile
    joblib.dump(accuracies, out_dir.joinpath('test_accuracies.pkl'))
