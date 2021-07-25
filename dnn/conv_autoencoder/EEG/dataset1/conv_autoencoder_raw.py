#! /bin/env/python3

import torch 
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from assess_eeg import assess_eeg, load_dnn_data
import matplotlib.pyplot as plt

# In this configuration of encoder we do not feed matrices of shape (subj, ims, features), 
# but raw eeg of shape (subj, ims, ch, time)
# This is not directly comparable to previous analyis where DNN got input of shape (subj, ims, features)


class dataset(torch.utils.data.Dataset):
    '''EEG dataset for training convolutional autoencoder.
    '''
    def __init__(self, data, normalize=True, tanh=False, scale=True):
        '''
        Inputs:
            data - numpy array or tensor of shape (subj, ims, chans, times)
            normalize - whether to normalize input to range to mean=0 and sd=1 along subjects. Default=True
            scale - whether to scale the data between -1 and 1 by dividing it into max of the abs of the values.
                Default=True
            tanh - whether to preprocess the data with Tanh function. Defualt=Fasle
        Methods:
            __getitem__ - returns array of shape (subj, chans, times) for the referenced image.
            __len__ - returns number of images 
            __len_subj__ - returns number of subjects
        '''
        with torch.no_grad():
            self.data = torch.tensor(data).permute(1,0,2,3).type(torch.float32) # to shape(ims, subjs, chans, times)
            if normalize:
                # normalize along DNN channels == subjects
                m = torch.mean(self.data, 1, keepdim=True)
                sd = torch.std(self.data, 1, keepdim=True, unbiased=False)
                self.data = (self.data - m)/sd
            # scale data between -1 and 1
            if tanh:
                tanh = torch.nn.Tanh()
                self.data = tanh(self.data)
            if scale:
                dat = torch.zeros_like(self.data)
                for subj in range(self.data.shape[1]):
                    dat[:,subj,:,:] = self.data[:,subj,:,:]/torch.max(torch.abs(self.data[:,subj,:,:]))
                self.data = dat

    def __len__(self):
        return self.data.shape[0]

    def __len_subj__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[idx, :, :, :]


class conv_autoencoder_raw(torch.nn.Module):
    '''
    NB! DNN works on raw EEG (subj, im, CH, TIME) insteas of (subj, im ,feat).
    Input shall be of shape (batch_size, subj, eeg_ch, eeg_time)
    Attributes:
        n_subj -int, n sunjects
        enc_chs, dec_chs - int, number of output channels for each conv layer of encoder
        p - float, dropout_probability. Default 0 (no dropout)
    Methods:
        forward. 
            Outputs:
            enc (ims, 1, feature1, feature2)
            dec (ims, subj, eeg_ch, eeg_time)
    '''
    def __init__(self, n_subj, enc_ch1=16, enc_ch2=32, enc_ch3=64,\
                dec_ch1 = 64, dec_ch2 = 32, \
                p=0):
         
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
        super(conv_autoencoder_raw, self).__init__()
        self.n_subj = n_subj

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        
        self.encoder = torch.nn.Sequential(conv1, conv2, conv3)
        self.decoder = torch.nn.Sequential(deconv1, deconv2, deconv3)

    def forward(self, data):
        orig_dims = data.shape[-2:]
        enc_out = self.encoder(data)
        dec_out = self.decoder(enc_out)
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
        projected_eeg_enc - array of shape (ims, features)
        projected_eeg_dec- array of shape (subj, ims, features)
    '''
    model.eval()
    projected_eeg_enc = []
    projected_eeg_dec = []
    for batch in dataloader: 
        enc, dec = model(batch)
        projected_eeg_enc.append(enc.cpu().detach().numpy())
        projected_eeg_dec.append(dec.cpu().detach().numpy())
    projected_eeg_enc = np.concatenate(projected_eeg_enc, axis=0)
    projected_eeg_enc = np.squeeze(projected_eeg_enc) # (ims, DDN_chs, chs, time)
    # average over DNN channels
    #projected_eeg_enc = np.mean(projected_eeg_enc, axis=1)
    # from (ims,chs, eeg_ch, eeg_time) to (ims, features)
    projected_eeg_enc = np.reshape(projected_eeg_enc, (projected_eeg_enc.shape[0],-1))

    projected_eeg_dec = np.concatenate(projected_eeg_dec, axis=0)
    # from (ims, subj, eeg_ch, eeg_time) to (subj, ims, eeg_ch, eeg_time)
    projected_eeg_dec = np.transpose(projected_eeg_dec, (1, 0, 2, 3))
    # from (subj, ims, eeg_ch, eeg_time) to (subj, ims, features) == flatten
    projected_eeg_dec = np.reshape(projected_eeg_dec, (projected_eeg_dec.shape[0],\
                                                        projected_eeg_dec.shape[1], -1))
    model.train()
    return projected_eeg_enc, projected_eeg_dec

def plot(batch, enc, dec, im=0, ignore_enc=False):
    batch_ = np.squeeze(batch.cpu().detach().numpy()[im])
    enc_ = np.mean(enc.cpu().detach().numpy()[im], axis=0)
    dec_ = np.squeeze(dec.cpu().detach().numpy()[im])

    if ignore_enc ==True:
        fig, ax = plt.subplots(2, sharey=True)
        min_ = min(np.min(batch_), np.mean(dec_))
        max_ = max(np.max(batch_),  np.max(dec_))
        im1=ax[0].imshow(batch_)
        im2=ax[1].imshow(dec_)
        for im in (im1, im2):
            im.set_clim(min_, max_)
        cbar1= fig.colorbar(im1)

    elif ingore_enc==False:
        fig, ax = plt.subplots(3, sharey=True)
        min_ = min(np.min(batch_), np.mean(enc_), np.mean(dec_))
        max_ = max(np.max(batch_), np.max(enc_), np.max(dec_))
        im1=ax[0].imshow(batch_)
        im2=ax[1].imshow(enc_)
        im3=ax[2].imshow(dec_)
        for im in (im1, im2, im3):
            im.set_clim(min_, max_)
        cbar1 = fig.colorbar(im1)
    return fig


if __name__=='__main__':
    import argparse
    import time
    import warnings
    from collections import defaultdict
    from pathlib import Path
    import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir',type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_raw/EEG/dataset1/draft/',
    help='/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/conv_autoencoder_raw/EEG/dataset1/draft/')
    parser.add_argument('-n_workers', type=int, default=0, help='default=0')
    parser.add_argument('-batch_size', type=int, default=16, help='default=16')
    parser.add_argument('-gpu', action='store_true', default=False, help='Flag, whether to '
    'use GPU. Default = False.')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate. Default=0.001')
    parser.add_argument('-n_epochs',type=int, default=10, help='How many times to pass '
    'through the dataset. Default=10')
    parser.add_argument('-bpl','--batches_per_loss',type=int, default=20, help='Save loss every '
    'bacth_per_loss mini-batches. Default=20.')
    parser.add_argument('-epta','--epochs_per_test_accuracy',type=int, default=1, help='Save test '
    'set accuracy every epochs_per_test_accuracy  epochs. Default == 1')
    parser.add_argument('-enc_chs',type=int, nargs=3, default= [16, 32, 64], help='Channels of encoder layer of DNN.')
    parser.add_argument('-dec_chs',type=int, nargs=2, default= [64, 32], help='Channels of decoder layer of DNN.')
    parser.add_argument('-p', type=int, default = 0, help = 'Dropout probability for encoder layer.')
    parser.add_argument('-normalize', type=int, default = 1, help = 'Bool (1/0), whether to normalize the data.'
    'Default=True.')
    parser.add_argument('-scale', type=int, default = 1, help = 'Bool (1/0), whether to scale the data '
    'between -1 and 1. Default=True.')
    parser.add_argument('-i','--interactive', action='store_true', default = False, help = 'Flag, whether to run the model in '
    'interactive mode. Default=False.')
    parser.add_argument('-eeg_dir', type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/50hz/time_window13-40/',\
    help='Directory with EEG dataset. Default=/scratch/akitaitsev/intersubject_generalization/linear/'
    'dataset_matrices/50hz/time_window13-40/')
    args = parser.parse_args()
    
    bpl = args.batches_per_loss
    epta = args.epochs_per_test_accuracy
    out_dir = Path(args.out_dir)

    # create datasets
    datasets_dir = Path(args.eeg_dir)
    data_train = joblib.load(datasets_dir.joinpath('dataset_train.pkl'))
    data_test = joblib.load(datasets_dir.joinpath('dataset_test.pkl'))

    dataset_train = dataset(data_train, normalize=bool(args.normalize), scale=bool(args.scale),\
        tanh=False)
    dataset_test = dataset(data_test, normalize=bool(args.normalize), scale=bool(args.scale),\
        tanh=False)


    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,\
                                                    shuffle=True, num_workers=args.n_workers,\
                                                    drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,\
                                                    shuffle=False, num_workers=args.n_workers,\
                                                    drop_last=False)

    train_dataloader_for_assessment = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,\
                                                    shuffle=False, num_workers=args.n_workers,\
                                                    drop_last=False)

    # Load DNN image activations for regression
    dnn_dir='/scratch/akitaitsev/encoding_Ale/dataset1/dnn_activations/'
    X_tr, X_val, X_test = load_dnn_data('CORnet-S', 1000, dnn_dir)

    # logging
    writer = SummaryWriter(out_dir.joinpath('runs'))

    # define the model
    model = conv_autoencoder_raw(n_subj = dataset_train.__len_subj__(), enc_ch1 = args.enc_chs[0], enc_ch2 = args.enc_chs[1],\
        enc_ch3 = args.enc_chs[2], dec_ch1 = args.dec_chs[0], dec_ch2 = args.dec_chs[1], p=args.p)

    if args.gpu and args.n_workers >=1:
        warnings.warn('Using GPU and n_workers>=1 can cause some difficulties.')
    if args.gpu:
        device_name="cuda"
        device=torch.device("cuda:0")
        model.to(device)
        model=torch.nn.DataParallel(model)
        print("Using "+str(torch.cuda.device_count()) +" GPUs.")
    elif not args.gpu:
        device=torch.device("cpu")
        device_name="cpu"
        print("Using CPU.")

    # define the loss
    loss_fn = torch.nn.MSELoss() 

    # define optmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Loss and  accuracy init
    losses = defaultdict()
    accuracies = defaultdict()
    accuracies["encoder"] = defaultdict()
    accuracies["encoder"]["average"] = []
    accuracies["decoder"] = defaultdict()
    accuracies["decoder"]["average"] = [] 
    accuracies["decoder"]["subjectwise"] = defaultdict()
    accuracies["decoder"]["subjectwise"]["mean"] = []
    accuracies["decoder"]["subjectwise"]["SD"] = []
    cntr_epta=0 

    figs = []

    # Loop through EEG dataset in batches
    for epoch in range(args.n_epochs):
        model.train()
        cntr=0
        tic = time.time()
        losses["epoch"+str(epoch)]=[]

        for batch in train_dataloader:
            if args.gpu:
                batch = batch.to(device)
            enc, dec = model.forward(batch)

            # compute loss - minimize diff between outputs of net and real data?
            loss = loss_fn(dec, batch) 
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
            
            if args.interactive:
                fig = plot(batch, enc, dec, im=0, ignore_enc=True)
                plt.show()
                inp = input('Next? y/n ')
                if inp=='y':
                    pass
                elif inp=='n':
                    break
                figs.append(fig)
            
            # Project train and test set EEG into new space
            eeg_train_proj_ENC, eeg_train_proj_DEC = project_eeg_raw(model, train_dataloader_for_assessment) 
            eeg_test_proj_ENC, eeg_test_proj_DEC = project_eeg_raw(model, test_dataloader)
            
            av_ENC = assess_eeg(X_tr, X_test, eeg_train_proj_ENC, eeg_test_proj_ENC, layer="encoder")
            av_DEC, sw_DEC = assess_eeg(X_tr, X_test, eeg_train_proj_DEC, eeg_test_proj_DEC)

            accuracies["encoder"]["average"].append(av_ENC[0])
            accuracies["decoder"]["average"].append(av_DEC[0])
            accuracies["decoder"]["subjectwise"]["mean"].append(sw_DEC[0])
            accuracies["decoder"]["subjectwise"]["SD"].append(sw_DEC[1])
            
            # Print info
            toc = time.time() - tic
            print('Network top1 generic decoding accuracy on encoder output at epoch {:d}:\n'
            'Average: {:.2f} %'.format(epoch, av_ENC[0]))
            print('Network top1 generic decoding accuracy on decoder output at epoch {:d}:\n'
            'Average: {:.2f} % \n'.format(epoch, av_DEC[0]))

            # logging 
            writer.add_scalar('accuracy_encoder_av', av_ENC[0],\
                    len(train_dataloader)*cntr_epta) 
            writer.add_scalar('accuracy_decoder_av', av_DEC[0],\
                    len(train_dataloader)*cntr_epta) 
        cntr_epta += 1
    writer.close()


    # Project EEG into new space using trained model
    projected_eeg = defaultdict()
    projected_eeg["train"] = defaultdict()
    projected_eeg["test"] = defaultdict() 
    projected_eeg["train"]["encoder"], _ = project_eeg_raw(model, train_dataloader)
    projected_eeg["test"]["encoder"], _ = project_eeg_raw(model, test_dataloader)
    
    # Create output dir
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
