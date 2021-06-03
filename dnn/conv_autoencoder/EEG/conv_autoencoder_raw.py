#! /bin/env/python3

import torch 
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MaxAbsScaler
from assess_eeg import assess_eeg
from assess_eeg import project_eeg

# In this configuration of encoder we do not feed matrices of shape (subj, ims, features), 
# but raw eeg of shape (subj, ims, ch, time)
# This is not directly comparable to previous analyis where DNN got input of shape (subj, ims, features)


class dataset(torch.utils.data.Dataset):
    '''EEG dataset for training convolutional autoencoder.
    '''
    def __init__(self, data, normalize=True):
        '''
        Inputs:
            data - numpy array or tensor of shape (subj, ims, chans, times)
            normalize - whether tio normalize input to range (-1,1). Default=True
        '''

        self.data = torch.tensor(data).permute(1,0,2,3) # to shape(ims, subjs, chans, times)
        if normalize:
            norm = torch.nn.BatchNorm2d(num_features = self.data.shape[1]) #n_subjects
            self.data = norm(self.data.type(torch.float32)).detach().numpy()
        else:
            self.data = self.data.type(torch.float32).detach().numpy()
        # scale data between -1 and 1
        scaler = MaxAbsScaler()
        dat = np.zeros_like(self.data)
        for im in range(self.data.shape[0]):
            for subj in range(self.data.shape[1]):
                dat[im, subj,:,:] = scaler.fit_transform(self.data[im,subj,:,:])
        self.data = torch.tensor(dat)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

class conv_autoencoder_raw(torch.nn.Module):
    '''
    NB! DNN works on raw EEG (subj, im, CH, TIME) insteas of (subj, im ,feat).
    Input shall be of shape (batch_size, subj, eeg_ch, eeg_time)
    Attributes:
        n_subj -int, n sunjects
        out_ch1, 2 ,3 - int, number of output channels for each conv layer of encoder
        feat_dims1, 2 ,3 - tuple of ints, feature dimensions at each conv layer
        p - float, dropout_probability. Default 0 (no dropout)
        no_resampling - bool, whether to implement resampling out output layers to
        spefified dimensions. If True, does not resample. Default=True.
        enc_layer -int, which encoder layer to resurn in the output. Default=2 (last layer)
    Methods:
        forward. 
            Outputs:
            enc (ims, 1, feature1, feature2)
            dec (ims, subj, eeg_ch, eeg_time)
    '''
    def __init__(self, n_subj, out_ch1=32, out_ch2=64, out_ch3=32,\
                feat_dims1=(8,12), feat_dims2 = (4,4), feat_dims3=(8,12), p=0,\
                no_resampling=True, enc_layer=2):
         
        conv1 = torch.nn.Sequential(torch.nn.Conv2d(n_subj, out_ch1, 3), \
                                torch.nn.BatchNorm2d(out_ch1),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        conv2 = torch.nn.Sequential(torch.nn.Conv2d(out_ch1, out_ch2, 3), \
                                torch.nn.BatchNorm2d(out_ch2),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        conv3 = torch.nn.Sequential(torch.nn.Conv2d(out_ch2, out_ch3, 3), \
                                torch.nn.BatchNorm2d(out_ch3),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        decoder = torch.nn.Sequential(torch.nn.Conv2d(out_ch3, n_subj, 3), \
                                torch.nn.BatchNorm2d(n_subj),\
                                torch.nn.Tanh())
        super(conv_autoencoder_raw, self).__init__()
        self.n_subj = n_subj
        self.out_ch1 = out_ch1
        self.out_ch2 = out_ch2
        self.out_ch3 = out_ch3
        self.feat_dims1 = feat_dims1 
        self.feat_dims2=feat_dims2
        self.feat_dims3=feat_dims3
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.no_resampling = no_resampling
        self.enc_layer = enc_layer

        self.decoder = decoder

    def forward(self, data):
        # encoder
        orig_dims = data.shape[-2:]
        batch_size = data.shape[0]
        out1 = self.conv1(data)
        if not self.no_resampling:
            out1 = F.interpolate(out1, self.feat_dims1)
        out2 = self.conv2(out1)
        if not self.no_resampling:
            out2 = F.interpolate(out2, self.feat_dims2)
        out3 = self.conv3(out2)
        if not self.no_resampling:
            out3 = F.interpolate(out3, self.feat_dims3)
        # decoder
        enc_layers=[out1, out2, out3]
        enc_out=enc_layers[self.enc_layer]
        dec_out = self.decoder(out3)
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
    # from (ims, 1, eeg_ch, eeg_time) to (ims, eeg_ch, eeg_time)
    projected_eeg_enc = np.squeeze(projected_eeg_enc) 
    # from (ims, eeg_ch, eeg_time) to (ims, features) == flatten
    projected_eeg_enc = np.reshape(projected_eeg_enc, (projected_eeg_enc.shape[0],-1))

    # from (ims, subj, eeg_ch, eeg_time) to (subj, ims, eeg_ch, eeg_time)
    projected_eeg_dec = np.concatenate(projected_eeg_dec, axis=0)
    projected_eeg_dec = np.transpose(projected_eeg_dec, (1, 0, 2, 3))
    # from (subj, ims, eeg_ch, eeg_time) to (subj, ims, features) == flatten
    projected_eeg_dec = np.reshape(projected_eeg_dec, (projected_eeg_dec.shape[0],\
                                                        projected_eeg_dec.shape[1], -1))
    model.train()
    return projected_eeg_enc, projected_eeg_dec


if __name__=='__main__':
    import argparse
    import time
    import warnings
    from collections import defaultdict
    from pathlib import Path
    import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir',type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/EEG/draft/',
    help='/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/EEG/draft/')
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
    parser.add_argument('-eeg_dir', type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/50hz/time_window13-40/',\
    help='Directory with EEG dataset. Default=/scratch/akitaitsev/intersubject_generalization/linear/'
    'dataset_matrices/50hz/time_window13-40/')
    parser.add_argument('-no_resampling', type=bool, default=True, help='Whether to resample output to specific '
    'dimensions. Default=True.')
    parser.add_argument('-enc_layer', type=int, default=2, help='Which layer of encoder to return as an output. Default=2.'
    'dimensions. Default=True.')
    args = parser.parse_args()
    
    bpl = args.batches_per_loss
    epta = args.epochs_per_test_accuracy
    out_dir = Path(args.out_dir)

    # create datasets
    datasets_dir = Path(args.eeg_dir)
    data_train = joblib.load(datasets_dir.joinpath('dataset_train.pkl'))
    data_test = joblib.load(datasets_dir.joinpath('dataset_test.pkl'))

    dataset_train = dataset(data_train)
    dataset_test = dataset(data_test)

    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,\
                                                    shuffle=True, num_workers=args.n_workers,\
                                                    drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,\
                                                    shuffle=False, num_workers=args.n_workers,\
                                                    drop_last=False)

    # logging
    writer = SummaryWriter(out_dir.joinpath('runs'))

    # define the model
    model = conv_autoencoder_raw(n_subj = 7, no_resampling = args.no_resampling, enc_layer=args.enc_layer)

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

    # Loop through EEG dataset in batches
    for epoch in range(args.n_epochs):
        model.train()
        cntr=0
        tic = time.time()
        losses["epoch"+str(epoch)]=[]

        for batch in train_dataloader:
            if args.gpu:
                batch = batch.to(device)
            enc, out = model.forward(batch)
            # compute loss - minimize diff between outputs of net and real data?
            loss = loss_fn(out, batch) 
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
        if cntr_epta % epta == 0 and epta!=0:
            tic = time.time()
            

            import matplotlib.pyplot as plt
            fig, axes=plt.subplots(3)
            axes[0].imshow(batch[0,0,:,:].cpu().detach().numpy())
            axes[1].imshow(enc[0,0,:,:].cpu().detach().squeeze().numpy())
            axes[2].imshow(out[0,0,:,:].cpu().detach().squeeze().numpy())
            import ipdb; ipdb.set_trace() 
            


            # Project train and test set EEG into new space
            eeg_train_proj_ENC, eeg_train_proj_DEC = project_eeg_raw(model, train_dataloader) 
            eeg_test_proj_ENC, eeg_test_proj_DEC = project_eeg_raw(model, test_dataloader)

            av_ENC = assess_eeg(eeg_train_proj_ENC, eeg_test_proj_ENC, layer="encoder")
            av_DEC, sw_DEC = assess_eeg(eeg_train_proj_DEC, eeg_test_proj_DEC)

            accuracies["encoder"]["average"].append(av_ENC[0])
            accuracies["decoder"]["average"].append(av_DEC[0])
            accuracies["decoder"]["subjectwise"]["mean"].append(sw_DEC[0])
            accuracies["decoder"]["subjectwise"]["SD"].append(sw_DEC[1])
            
            # Print info
            toc = time.time() - tic
            print('Network top1 generic decoding accuracy on encoder output at epoch {:d}:\n'
            'Average: {:.2f} %'.format(epoch, av_ENC[0]))
            print('Network top1 generic decoding accuracy on decoder output at epoch {:d}:\n'
            'Average: {:.2f} % \n Subjectwise: {:.2f} % +- {:.2f} (SD)'.format(epoch, av_DEC[0], sw_DEC[0], sw_DEC[1]))
            print('Elapse time: {:.2f} minutes.'.format(toc/60))   

            # logging 
            writer.add_scalar('accuracy_encoder_av', av_ENC[0],\
                    len(train_dataloader)*cntr_epta) 
            writer.add_scalar('accuracy_decoder_av', av_DEC[0],\
                    len(train_dataloader)*cntr_epta) 
            writer.add_scalar('accuracy_decoder_sw', sw_DEC[0],\
                    len(train_dataloader)*cntr_epta) 
        cntr_epta += 1
    writer.close()


    # Project EEG into new space using trained model
    projected_eeg = defaultdict()
    projected_eeg["train"] = defaultdict()
    projected_eeg["test"] = defaultdict() 
    projected_eeg["train"]["encoder"], projected_eeg["train"]["decoder"] = project_eeg_raw(model, train_dataloader)
    projected_eeg["test"]["encoder"], projected_eeg["test"]["decoder"] = project_eeg_raw(model, test_dataloader)
    
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
