#! /bin/env/python3

# Implementation of conv autoencoder isnpired by the architechture of
# U-net

import numpy as np
import torch 
import torch.nn.functional as F
from assess_eeg import assess_eeg, load_dnn_data
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import interpolate


class dataset(torch.utils.data.Dataset):
    '''EEG dataset for training convolutional autoencoder.
    '''
    def __init__(self, data):
        '''
        Inputs:
            data - numpy array or tensor of shape (subj,ims, chans, times)
        '''
        self.data = torch.tensor(data).permute(1,0,2,3) # to shape(ims, subjs, chans, times)
        if torch.cuda.is_available():
            self.data = self.data.cuda()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].type(torch.float32)


class conv_autoencoder(torch.nn.Module):
    '''
    Input shall be of shape (batch_size(ims), subj, eeg_ch, eeg_time)
    Attributes:
        n_subj -int, n sunjects
        ch_enc, ch_dec - ints, encoder and decoder blocks output channels. 
        Default values:
            ch_enc1 = 32
            ch_enc2 = 64
            ch_enc3 = 128
            ch_enc4 = 256
            ch_enc5 = 512
            enc_out = 512
            ch_dec1 = 256
            ch_dec2 = 128
            ch_dec3 = 64
            ch_dec4 = 32
        p_enc - dropout value for encoder blocks 1-4
        out_dims - tuple of ch*time output dimensions of decoder. Shall match EEG dimensions.
            Default = (17,23) for 50hz data
    Methods:
        forward: returns encoder_out, decoder_out
        encoder_out - shape (subj, features)
        decoder_out - shape (subj, ch, time) - same as original EEG
    '''

    def __init__(self, n_subj, ch_time = (17,23),\
        ch_enc1 = 16, ch_enc2 = 32, ch_enc3 = 64, ch_enc4 = 128, ch_enc5 = 256,\
        dims_enc = (None, None, None, None, None),\
        paddings_enc = (1,1,1,1,1),\
        enc_out = 256,\
        ch_dec1 = 128, ch_dec2 = 64, ch_dec3 = 32, ch_dec4 = 16,\
        dims_dec = (None, None, None, None),\
        paddings_dec = (2,2,2,2,2),\
        p_enc=0):

        ## encoder
        # block 1
        conv11e = torch.nn.Sequential(torch.nn.Conv2d(n_subj, ch_enc1, (3,3), padding=paddings_enc[0]), \
                                torch.nn.BatchNorm2d(ch_enc1),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p_enc))
        conv12e = torch.nn.Sequential(torch.nn.Conv2d(ch_enc1, ch_enc1, (3,3), padding=paddings_enc[0]), \
                                torch.nn.BatchNorm2d(ch_enc1),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p_enc))
        pool1e = torch.nn.MaxPool2d((2,2))
        block1e = torch.nn.Sequential(conv11e, conv12e, pool1e)

        # block 2
        conv21e = torch.nn.Sequential(torch.nn.Conv2d(ch_enc1, ch_enc2, (3,3), padding=paddings_enc[1] ), \
                                torch.nn.BatchNorm2d(ch_enc2),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p_enc))
        conv22e = torch.nn.Sequential(torch.nn.Conv2d(ch_enc2, ch_enc2, (3,3), padding=paddings_enc[1]), \
                                torch.nn.BatchNorm2d(ch_enc2),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p_enc))
        pool2e = torch.nn.MaxPool2d((2,2))
        block2e = torch.nn.Sequential(conv21e, conv22e, pool2e)
    
        # block 3
        conv31e = torch.nn.Sequential(torch.nn.Conv2d(ch_enc2, ch_enc3, (3,3), padding=paddings_enc[2]), \
                                torch.nn.BatchNorm2d(ch_enc3),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p_enc))
        conv32e = torch.nn.Sequential(torch.nn.Conv2d(ch_enc3, ch_enc3, (3,3), padding=paddings_enc[2]), \
                                torch.nn.BatchNorm2d(ch_enc3),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p_enc))
        pool3e = torch.nn.MaxPool2d((2,2))
        block3e = torch.nn.Sequential(conv31e, conv32e)#, pool3e)

        # block 4
        conv41e = torch.nn.Sequential(torch.nn.Conv2d(ch_enc3, ch_enc4, (3,3), padding=paddings_enc[3]), \
                                torch.nn.BatchNorm2d(ch_enc4),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p_enc))
        conv42e = torch.nn.Sequential(torch.nn.Conv2d(ch_enc4, ch_enc4, (3,3), padding=paddings_enc[3]), \
                                torch.nn.BatchNorm2d(ch_enc4),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p_enc))
        pool4e = torch.nn.MaxPool2d((2,2))
        block4e = torch.nn.Sequential(conv41e, conv42e)#, pool4e)

        # block5 - final output
        enc1 = torch.nn.Sequential(torch.nn.Conv2d(ch_enc4, ch_enc5, (3,3), padding=paddings_enc[4]), \
                                torch.nn.BatchNorm2d(ch_enc5),\
                                torch.nn.ReLU())
        enc2 = torch.nn.Sequential(torch.nn.Conv2d(ch_enc5, enc_out, (3,3), padding=paddings_enc[4]), \
                                torch.nn.BatchNorm2d(enc_out),\
                                torch.nn.ReLU())
        block5e = torch.nn.Sequential(enc1, enc2)


        ### decoder
        # block1
        up_conv1d= torch.nn.Sequential(torch.nn.ConvTranspose2d(enc_out, ch_dec1, (2,2), padding=paddings_dec[0]), \
                                torch.nn.BatchNorm2d(ch_dec1),\
                                torch.nn.ReLU())
        conv11d= torch.nn.Sequential(torch.nn.Conv2d(ch_dec1, ch_dec1, (3,3), padding=paddings_dec[0]), \
                                torch.nn.BatchNorm2d(ch_dec1),\
                                torch.nn.ReLU())
        conv12d= torch.nn.Sequential(torch.nn.Conv2d(ch_dec1, ch_dec1, (3,3), padding=paddings_dec[0]), \
                                torch.nn.BatchNorm2d(ch_dec1),\
                                torch.nn.ReLU())
        block1d = torch.nn.Sequential(up_conv1d, conv11d, conv12d)
        # block2
        up_conv2d= torch.nn.Sequential(torch.nn.ConvTranspose2d(ch_dec1, ch_dec2, (2,2), padding=paddings_dec[1]), \
                                torch.nn.BatchNorm2d(ch_dec2),\
                                torch.nn.ReLU())
        conv21d= torch.nn.Sequential(torch.nn.Conv2d(ch_dec2, ch_dec2, (3,3), padding=paddings_dec[1]), \
                                torch.nn.BatchNorm2d(ch_dec2),\
                                torch.nn.ReLU())
        conv22d= torch.nn.Sequential(torch.nn.Conv2d(ch_dec2, ch_dec2, (3,3), padding=paddings_dec[1]), \
                                torch.nn.BatchNorm2d(ch_dec2),\
                                torch.nn.ReLU())
        block2d = torch.nn.Sequential(up_conv2d, conv21d, conv22d)
        # block3
        up_conv3d= torch.nn.Sequential(torch.nn.ConvTranspose2d(ch_dec2, ch_dec3, (2,2), padding=paddings_dec[2]), \
                                torch.nn.BatchNorm2d(ch_dec3),\
                                torch.nn.ReLU())
        conv31d= torch.nn.Sequential(torch.nn.Conv2d(ch_dec3, ch_dec3, (3,3), padding=paddings_dec[2]), \
                                torch.nn.BatchNorm2d(ch_dec3),\
                                torch.nn.ReLU())
        conv32d= torch.nn.Sequential(torch.nn.Conv2d(ch_dec3, ch_dec3, (3,3), padding=paddings_dec[2]), \
                                torch.nn.BatchNorm2d(ch_dec3),\
                                torch.nn.ReLU())
        block3d = torch.nn.Sequential(up_conv3d, conv31d, conv32d)

        # block4
        up_conv4d= torch.nn.Sequential(torch.nn.ConvTranspose2d(ch_dec3, ch_dec4, (2,2), padding=paddings_dec[3]), \
                                torch.nn.BatchNorm2d(ch_dec4),\
                                torch.nn.ReLU())
        conv41d= torch.nn.Sequential(torch.nn.Conv2d(ch_dec4, ch_dec4, (3,3), padding=paddings_dec[3]), \
                                torch.nn.BatchNorm2d(ch_dec4),\
                                torch.nn.ReLU())
        conv42d= torch.nn.Sequential(torch.nn.Conv2d(ch_dec4, ch_dec4, (3,3), padding=paddings_dec[3]), \
                                torch.nn.BatchNorm2d(ch_dec4),\
                                torch.nn.ReLU())
        block4d = torch.nn.Sequential(up_conv4d, conv41d, conv42d)

        # block5 - final output
        block5d = torch.nn.Sequential(torch.nn.Conv2d(ch_dec4, n_subj, (1,1), padding=paddings_dec[4]), \
                                torch.nn.BatchNorm2d(n_subj),\
                                torch.nn.Sigmoid())

        encoder = torch.nn.Sequential(block1e, block2e, block3e, block4e, block5e)
        decoder = torch.nn.Sequential(block1d, block2d, block3d, block4d, block5d)

        super(conv_autoencoder, self).__init__()

        self.block1e=block1e
        self.block2e=block2e
        self.block3e=block3e
        self.block4e=block4e
        self.block5e = block5e
        self.block1d=block1d
        self.block2d=block2d
        self.block3d=block3d
        self.block4d = block4d
        self.block5d = block5d
        
        self.n_subj = n_subj
        self.ch_time = ch_time

        self.dims_enc = dims_enc
        self.dims_dec = dims_dec

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        # encoder
        e1 = self.block1e(data)
        if not self.dims_enc[0] == None:
            e1 = interpolate(e1, self.dims_enc[0])
        e2=self.block2e(e1)
        if not self.dims_enc[1] == None:
            e2 = interpolate(e2, self.dims_enc[1])
        e3=self.block3e(e2)
        if not self.dims_enc[2] == None:
            e3 = interpolate(e3, self.dims_enc[2])
        e4=self.block4e(e3)
        if not self.dims_enc[3] == None:
            e4 = interpolate(e4, self.dims_enc[3])
        e5=self.block5e(e4)
        if not self.dims_enc[4] == None:
            e5 = interpolate(e5, self.dims_enc[4])
        
        # decoder
        d1=self.block1d(e5.float())
        if not self.dims_dec[0] == None:
            d1 = interpolate(d1, self.dims_dec[0])
        d2=self.block2d(d1)
        if not self.dims_dec[1] == None:
            d2 = interpolate(d2, self.dims_dec[1])
        d3=self.block3d(d2)
        if not self.dims_dec[2] == None:
            d3 = interpolate(d3, self.dims_dec[2])
        d4=self.block4d(d3)
        if not self.dims_dec[3] == None:
            d4 = interpolate(d4, self.dims_dec[3])
        d5=self.block5d(d4)
        if not (d5.shape[-2], d5.shape[-1]) == ch_time:
            d5 = interpolate(d5, ch_time)
        # e5.shape = (batch_size, ch, time)
        # d5.shape = (batch_size, subj, ch, time)
        e5 = torch.reshape(e5, (e5.shape[0], -1)) #(batch_size, features) 
        return  e5, d5


def project_eeg_conv_autoenc(model, dataloader):
    '''
    Project EEG into new space independently for every subject using 
    trained model.
    Inputs:
        model - trained conv_autoenc model 
        dataloader - dataloader for eeg_dataset_test class instance.
        layer - str, encoder or proj_head. Outputs of which layer to treat as
                projected EEG. Default = "proj_head".
    Ouputs:
        projected_eeg - 2d numpy array of shape (ims, features) 
                        of eeg projected into new (shared) space.
    '''
    model.eval()
    projected_eeg = []
    for data in dataloader:
        if torch.cuda.is_available():
            data=data.cuda()    
        feature, out = model(data)
        projected_eeg.append(feature.cpu().detach().numpy())
    projected_eeg = np.concatenate(projected_eeg, axis=0)
    model.train()
    return projected_eeg



if __name__=='__main__':
    import argparse
    import time
    import warnings
    from collections import defaultdict
    from pathlib import Path
    import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument('-out_dir',type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/EEG/dataset1/draft/',
    help='/scratch/akitaitsev/intersubject_generalization/dnn/conv_autoencoder/EEG/dataset1/draft/')
    parser.add_argument('-eeg_dir',type=str, default=\
    '/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/50hz/time_window13-40/',
    help='/scratch/akitaitsev/intersubject_generalization/linear/dataset1/dataset_matrices/50hz/time_window13-40/')
    parser.add_argument('-n_workers', type=int, default=0, help='default=0')
    parser.add_argument('-batch_size', type=int, default=16, help='default=16')
    parser.add_argument('-gpu', action='store_true', default=False, help='Flag, whether to '
    'use GPU. Default = False.')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate. Default=0.001')
    parser.add_argument('-n_epochs',type=int, default=10, help='How many times to pass '
    'through the dataset. Default=10')
    parser.add_argument('-bpl', type=int, default=20, help='Save loss every '
    'bacth_per_loss mini-batches. Default=20.')
    parser.add_argument('-epta', type=int, default=1, help='Save test '
    'set accuracy every epochs_per_test_accuracy  epochs. Default == 1')
    parser.add_argument('-enc_chs', nargs=5, type=int, default = [16, 32, 64, 128, 256], help=\
    'output channels for encoder blocks 1-5. Default = [16, 32, 64, 128, 256].')
    parser.add_argument('-enc_out', type=int, default= 256, help='encoder final layer '
    'output channels. Default=512.')
    parser.add_argument('-dec_chs', nargs=4, type=int, default=[128, 64, 32, 16], help=\
    'output channels for decoder blocks 1-4. Default = [128, 64, 32, 16].')
    #parser.add_argument('-dims_enc', nargs=5, type=int, default=[None, None, None, None,None], help=\
    #'Output dimensions of encoder layers. Default = Nones, no interpolation.')
    #parser.add_argument('-dims_dec', nargs=4, type=int, default=[None, None, None, None], help=\
    #'Output dimensions of decoder layers. Default = Nones, no interpolation.')
    parser.add_argument('-paddings_enc', nargs=5, type=int, default=[1,1,1,1,1], help=\
    'Size of paddings for encoder layers. Default=[1,1,1,1,1].')
    parser.add_argument('-paddings_dec', nargs=5, type=int, default=[2,2,2,2,2], help=\
    'Size of paddings for decoder layers. Default=[2,2,2,2,2].')
    parser.add_argument('-p_enc', type=float, default=0., help='Dropout value for encoder blocks 1-4. '
    'Default=0.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # create datasets
    # EEG datasets
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
    # no shuffling
    train_dataloader_for_assessment = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,\
                                                    shuffle=False, num_workers=args.n_workers,\
                                                    drop_last=False)

    # Load DNN image activations for regression
    dnn_dir='/scratch/akitaitsev/encoding_Ale/dataset1/dnn_activations/'
    X_tr, X_val, X_test = load_dnn_data('CORnet-S', 1000, dnn_dir)

    # logging
    writer = SummaryWriter(out_dir.joinpath('runs'))

    # define the model
    #dims_enc=([15,20],[12,17],[7,12], [5,8],[3,5])
    #dims_dec=(None, None, None, None, None)
    ch_time = (data_test.shape[-2], data_test.shape[-1])

    model = conv_autoencoder(n_subj = data_test.shape[0], ch_time = ch_time,\
        ch_enc1=args.enc_chs[0], ch_enc2 = args.enc_chs[1], ch_enc3 =  args.enc_chs[2],\
        ch_enc4 =  args.enc_chs[3], ch_enc5 = args.enc_chs[4], \
        enc_out = args.enc_out,\
        #dims_enc = dims_enc,\
        ch_dec1 = args.dec_chs[0], ch_dec2 = args.dec_chs[1],\
        ch_dec3 = args.dec_chs[2], ch_dec4 = args.dec_chs[3],\
        #dims_dec = dims_dec,\
        p_enc=args.p_enc)
        
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
    accuracies["encoder"]["subjectwise"] = defaultdict()
    accuracies["encoder"]["subjectwise"]["mean"] = []
    accuracies["encoder"]["subjectwise"]["SD"] = []
    cntr_epta=0 

    # Loop through EEG dataset in batches
    for epoch in range(args.n_epochs):
        model.train()
        cntr=0
        tic = time.time()
        losses["epoch"+str(epoch)]=[]
        accuracies["epoch"+str(epoch)]=[]

        for batch in train_dataloader:
            if args.gpu:
                batch = batch.to(device)
            enc, dec = model.forward(batch)#[:,0,:,:].unsqueeze(1))
           
            # compute loss - minimize diff between outputs of net and real data?
            loss = loss_fn(dec, batch)#[:,0,:,:].unsqueeze(1)) 
            losses["epoch"+str(epoch)].append(loss.cpu().detach().numpy())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # save loss every bpl mini-batches
            if cntr % args.bpl == 0 and cntr != 0:
                writer.add_scalar('training_loss', sum(losses["epoch"+str(epoch)][-args.bpl:])/args.bpl, cntr+\
                len(train_dataloader)*epoch) 
                toc=time.time() - tic
                tic=time.time()
                print('Epoch {:d}, average loss over bacthes {:d} - {:d}: {:.3f}. Time, s: {:.1f}'.\
                format(epoch, int(cntr-args.bpl), cntr, sum(losses["epoch"+str(epoch)][-args.bpl:])/args.bpl, toc))
            cntr+=1

        # save test accuracy every epta epcohs
        if cntr_epta % args.epta == 0 and cntr_epta !=0:
            tic = time.time()
            
            # Project train and test set EEG into new space

            # treat encoder output as EEG
            eeg_train_projected_ENC = project_eeg_conv_autoenc(model, train_dataloader_for_assessment)
            eeg_test_projected_ENC = project_eeg_conv_autoenc(model, test_dataloader)
            av_ENC, sw_ENC = assess_eeg(X_tr, X_test, eeg_train_projected_ENC, eeg_test_projected_ENC, layer="encoder")

            accuracies["encoder"]["average"].append(av_ENC)

            print('Network top1 generic decoding accuracy on encoder output at epoch {:d}:\n'
                'Average: {:.2f} % \n)'.format(epoch, av_ENC))
            print('Elapse time: {:.2f} minutes.'.format(toc/60))   

            # logging 
            writer.add_scalar('accuracy_encoder_av', av_ENC,\
                    len(train_dataloader)*cntr_epta) 

        cntr_epta += 1
    writer.close()

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    # Project EEG into new space using trained model
    projected_eeg = defaultdict()
    projected_eeg["train"] = defaultdict()
    projected_eeg["test"] = defaultdict() 
    projected_eeg["train"]["encoder"] = project_eeg_conv_autoenc(model, train_dataloader)
    projected_eeg["test"]["encoder"] = project_eeg_conv_autoenc(model, test_dataloader) 
    
    # save projected EEG 
    joblib.dump(projected_eeg, out_dir.joinpath('projected_eeg.pkl'))

    # save trained model
    torch.save(model.state_dict(), out_dir.joinpath('trained_model.pt'))

    # save loss profile
    joblib.dump(losses, out_dir.joinpath('losses.pkl')) 
    
    # save test accuracy profile
    joblib.dump(accuracies, out_dir.joinpath('test_accuracies.pkl'))
