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


class conv_autoencoder_raw_CL(torch.nn.Module):
    '''
    conv autoencoder for itraining with Contrastive Loss.
    Input shall be of shape (batch_size=ims, subj, eeg_ch, eeg_time)

    Attributes:
        n_subj -int, n subjects in one batch. Default =7
        enc_chs, dec_chs - int, number of output channels for each conv layer of encoder
        p - float, dropout_probability. Default 0 (no dropout)
        interpolate - bool, whether to interpolate decoder output to the same dims as ch, time in input.
        Default=True.
    Methods:
        forward. 
            Outputs:
            enc (ims, DNN_chs_enc, feature1, feature2)
            dec (ims, DNN_chs_dec, feature1, feature2)
    '''
    def __init__(self, n_subj=7, enc_ch1=16, enc_ch2=32, enc_ch3=64,\
                dec_ch1 = 32, dec_ch2 = 16, \
                p=0, interpolate=True):
         
        conv1 = torch.nn.Sequential(torch.nn.Conv2d(n_subj, enc_ch1, 3), \
                                torch.nn.BatchNorm2d(enc_ch1),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        conv2 = torch.nn.Sequential(torch.nn.Conv2d(enc_ch1, enc_ch2, 3), \
                                torch.nn.BatchNorm2d(enc_ch2),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        conv3 = torch.nn.Sequential(torch.nn.Conv2d(enc_ch2, enc_ch3, 3), \
                                torch.nn.BatchNorm2d(enc_ch3),\
                                torch.nn.Tanh())
        deconv1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(enc_ch3, dec_ch1, 3), \
                                torch.nn.BatchNorm2d(dec_ch1),\
                                torch.nn.ReLU())
        deconv2 = torch.nn.Sequential(torch.nn.ConvTranspose2d(dec_ch1, dec_ch2, 3), \
                                torch.nn.BatchNorm2d(dec_ch2),\
                                torch.nn.ReLU())
        deconv3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(dec_ch2, n_subj, 3), \
                                torch.nn.BatchNorm2d(n_subj),\
                                torch.nn.Tanh())
        super(conv_autoencoder_raw_CL, self).__init__()
        self.n_subj = n_subj
        self.interpolate = interpolate

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        
        self.encoder = torch.nn.Sequential(conv1, conv2, conv3)
        self.decoder = torch.nn.Sequential(deconv1, deconv2, deconv3)

    def forward(self, data):
        enc_out = self.encoder(data)
        dec_out = self.decoder(enc_out)
        if self.interpolate:
            orig_dims = data.shape[-2:]
            if not dec_out[-2:] == orig_dims:
                dec_out = F.interpolate(dec_out, orig_dims)
        return enc_out, dec_out


def project_eeg_raw(model, dataloader):
    '''Project EEG into new space using DNN model.
    Inputs:
        dataloader - dataloader which yields batches of size
                     (ims, subj, eeg_ch, eeg_time)
        model - DNN which returns encoder and out and accepts
                input of shape (ims, subj, eeg_ch, eeg_time)
    Ouputs:
        eeg_enc - array of shape (ims, features)
        eeg_dec- array of shape (ims, subjs, features)
    '''
    model.eval()
    eeg_enc = []
    eeg_dec = []
    for batch in dataloader: 
        enc, dec = model(batch) # (im_per_batch, n_subj, ch, time)
        eeg_enc.append(enc.cpu().detach().numpy())
        eeg_dec.append(dec.cpu().detach().numpy())

    eeg_enc = np.concatenate(eeg_enc, axis=0)  
    eeg_dec = np.concatenate(eeg_dec, axis=0)

    eeg_enc = np.reshape(eeg_enc, (eeg_enc.shape[0], -1)) # (ims, features)
    eeg_dec = np.reshape(eeg_dec, (eeg_dec.shape[0], eeg_dec.shape[1],-1)) # (im, subj, features)
    eeg_dec = np.transpose(eeg_dec, (1,0,2)) # (subj, ims, features)

    model.train()
    return eeg_enc, eeg_dec 



class dataset_conv_autoencoder_CL(torch.utils.data.Dataset):
    '''EEG dataset for tarining DNNs with contrasttive loss.
    Returns EEG response of 2 randomly picked 
    non-overlapping subjects. 
    Methods:
    __init__:
        Inputs:
        eeg_dataset - numpy array of eeg dataset of 
            shape (subj, ims, chans, times)
        transformer - transformer to apply to the 
                      eeg dataset. If None, converts 
                      eeg_dataset to tensor
            getitem returns batches of shape

    __getitem__:
        Inputs:
            idx - index along image dimension
        Outputs:
            (ims, subjs, ch, time, eeg_chan)

    __len__: return number of images

    __len_subj__: return number of subjects
    '''

    def __init__(self, eeg_dataset, transform = None, normalize=False, scale=False):
        if transform == None:
            self.eeg = torch.tensor(eeg_dataset)
        else:
            self.eeg = self.transformer(eeg_dataset)

        with torch.no_grad():
            if normalize:
                # normalize along DNN channels == subjects
                m = torch.mean(self.data, 0, keepdim=True)
                sd = torch.std(self.data, 0, keepdim=True, unbiased=False)
                self.data = (self.data - m)/sd
            # scale data between -1 and 1
            if scale:
                dat = torch.zeros_like(self.data)
                for subj in range(self.data.shape[0]):
                    dat[subj,:,:,:] = self.data[subj,:,:,:]/torch.max(torch.abs(self.data[subj,:,:,:]))
                self.data = dat

    def __len__(self):
        '''Return number of images'''
        return self.eeg.shape[1]

    def __getitem__(self, idx):
        '''
        idx - index along images
        '''
        subj_idx = np.random.permutation(np.linspace(0, self.eeg.shape[0],\
            self.eeg.shape[0], endpoint=False, dtype=int))
        batch = self.eeg[:,idx,:,:].type(torch.float32) # all subjects with index idx
        return batch



def ContrastiveLoss_leftthomas(out1, out2, batch_size, temperature, normalize="normalize"):
    '''Inputs:
        out1, out2 - outputs of dnn input to which were batches of images yielded
                     from eeg_dataset_train 
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


    
if __name__=='__main__':
    import argparse
    import time
    import warnings
    import copy
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir',type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_CL/EEG/dataset1/',
    help='/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_CL/EEG/dataset1/')
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
    parser.add_argument('-clip_grad_norm', type=int, default=None, help='Int, value for gradient clipping by norm. Default=None, '
    'no clipping.')
    parser.add_argument('-pick_best_net_state',  action='store_true', default=False, help='Flag, whether to pick up the model with best '
    'generic decoding accuracy on encoder projection head layer over epta epochs to project the data. If false, uses model at '
    'the last epoch to project dadta. Default=False.')

    parser.add_argument('-enc_chs',type=int, nargs=3, default= [8, 16, 32], help='Channels of encoder layer of DNN.')
    parser.add_argument('-dec_chs',type=int, nargs=2, default= [32, 16], help='Channels of decoder layer of DNN.')
    parser.add_argument('-p', type=int, default = 0, help = 'Dropout probability for encoder layer.')
    parser.add_argument('-normalize', type=int, default = 1, help = 'Bool (1/0), whether to normalize the data.'
    'Default=True.')
    parser.add_argument('-scale', type=int, default = 1, help = 'Bool (1/0), whether to scale the data '
    'between -1 and 1. Default=True.')
    args=parser.parse_args()


    bpl = args.batches_per_loss
    epta = args.epochs_per_test_accuracy
    out_dir = Path(args.out_dir)

    # EEG datasets
    datasets_dir = Path(args.eeg_dir)
    data_train = joblib.load(datasets_dir.joinpath('dataset_train.pkl'))
    data_test = joblib.load(datasets_dir.joinpath('dataset_test.pkl'))
   
    dataset_train = dataset_conv_autoencoder_CL(data_train)
    dataset_test = dataset_conv_autoencoder_CL(data_test)

    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,\
                                                shuffle=True, num_workers=args.n_workers,\
                                                drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, \
                                                shuffle = False, num_workers=args.n_workers,\
                                                drop_last=False)
    # no shuffling
    train_dataloader_for_assessment = torch.utils.data.DataLoader(dataset_train,\
                                                batch_size=args.batch_size, shuffle = False, num_workers=args.n_workers,\
                                                drop_last=False)

    # DNN data for regression
    dnn_dir='/scratch/akitaitsev/encoding_Ale/dataset1/dnn_activations/'
    X_train, X_val, X_test = load_dnn_data('CORnet-S', 1000, dnn_dir)

    # logging
    writer = SummaryWriter(out_dir.joinpath('runs'))    

    # define the model
    n_subj = data_train.shape[0]
    model = conv_autoencoder_raw_CL(n_subj=n_subj, enc_ch1=args.enc_chs[0], enc_ch2 = args.enc_chs[1],\
        enc_ch3 = args.enc_chs[2], dec_ch1 = args.dec_chs[0], dec_ch2 = args.dec_chs[1], p=args.p)


    if args.gpu and args.n_workers >=1:
        warnings.warn('Using GPU and n_workers>=1 can cause some difficulties.')
    if args.gpu:
        device_name="cuda"
        device=torch.device("cuda:0")
        model.to(torch.device("cuda:0"))
        model=torch.nn.DataParallel(model) 
        print("Using "+str(torch.cuda.device_count()) +" GPUs.")
    elif not args.gpu: 
        device=torch.device("cpu")
        device_name="cpu"
        print("Using CPU.") 
    
    # define optmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    # Loss and  accuracy init
    losses = defaultdict()
    accuracies = defaultdict()
    accuracies["encoder"] = []
    accuracies["decoder"] = defaultdict()
    accuracies["decoder"]["average"] = []
    accuracies["decoder"]["subjectwise"] = defaultdict()
    accuracies["decoder"]["subjectwise"]["mean"]=[]
    accuracies["decoder"]["subjectwise"]["SD"]=[]

    cntr_epta=0 
    net_states=[]

    # Loop through EEG dataset in batches
    for epoch in range(args.n_epochs):
        model.train()
        cntr=0
        tic = time.time()
        losses["epoch"+str(epoch)]=[]
        accuracies["epoch"+str(epoch)]=[]

        for batch in train_dataloader:
            if args.gpu:
                batch = batch.cuda()
            enc, dec = model.forward(batch)

            # select subject indices m and n so that m!=n
            subj_idx = np.random.permutation(np.linspace(0, n_subj,\
                n_subj, endpoint=False, dtype=int))
            n = subj_idx[0]
            m = subj_idx[1]

            # separate dec output from subject m and n where n!=m
            dec1 = torch.reshape(dec[:,n,:,:], (dec.shape[0],-1)) # (ims, features) from subject n
            dec2 = torch.reshape(dec[:,m,:,:], (dec.shape[0],-1)) # (ims, features) from subject m

            # compute loss
            loss = ContrastiveLoss_leftthomas(dec1, dec2, args.batch_size, args.temperature)
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
            
            # encoder - (ims, features), decoder - (subj, ims, features)
            train_enc, train_dec = project_eeg_raw(model, train_dataloader_for_assessment) 
            test_enc, test_dec = project_eeg_raw(model, test_dataloader)
            av_ENC, sw_ENC = assess_eeg(X_tr, X_test, train_enc, test_enc, layer='encoder')
            av_DEC, sw_DEC = assess_eeg(X_tr, X_test, train_dec, test_dec, layer='decoder')
            
            accuracies["encoder"].append(av_ENC)
            accuracies["decoder"]["average"].append(av_DEC[0])
            accuracies["decoder"]["subjectwise"]["mean"].append(sw_DEC[0])
            accuracies["decoder"]["subjectwise"]["SD"].append(sw_DEC[1])
            
            # Print info
            toc = time.time() - tic
            print('Network top1 generic decoding accuracy on encoder output at epoch {:d}:\n'
            '{:.2f} %'.format(epoch, av_ENC))
            print('Network top1 generic decoding accuracy on decoder output at epoch {:d}:\n'
            'Average: {:.2f} % \nSubjectwise: {:.2f} % +- {:.2f} (SD)'.format(epoch, av_DEC[0], sw_DEC[0], sw_DEC[1]))
            print('Elapse time: {:.2f} minutes.'.format(toc/60))   

            # logging 
            writer.add_scalar('accuracy_encoder', av_ENC,\
                    len(train_dataloader)*cntr_epta) 
            writer.add_scalar('accuracy_decoder_av', av_DEC[0],\
                    len(train_dataloader)*cntr_epta) 
            writer.add_scalar('accuracy_decoder_sw', sw_DEC[0],\
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
    projected_eeg["train"]["encoder"], projected_eeg["train"]["decoder"]  = project_eeg_raw(model, train_dataloader_for_assessment)
    projected_eeg["test"]["encoder"], projected_eeg["test"]["decoder"] = project_eeg(model, test_dataloader)
    
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
