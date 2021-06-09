#! /bin/env/python3

import torch 
import torch.nn.functional as F
from assess_eeg import assess_eeg
from assess_eeg import project_eeg

# In this configuration of encoder we do not feed matrices of shape (subj, ims, features), 
# but raw eeg of shape (subj, ims, ch, time)
# This is not directly comparable to previous analyis where DNN got input of shape (subj, ims, features)


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
            return self.data[idx]

class conv_autoencoder(torch.nn.Module):
    '''
    Input shall be of shape (batch_size(ims), subj, eeg_ch, eeg_time)
    Attributes:
        n_subj -int, n sunjects
        out_ch1, 2 ,3 - int, number of output channels for each conv layer of encoder
        feat_dims1, 2 ,3 - tuple of ints, feature dimensions at each conv layer
        p - float, dropout_probability. Default 0.5
    '''
    def __init__(self, n_subj, out_ch1=32, out_ch2=16, out_ch3=1,\
                feat_dims1=(8,12), feat_dims2 = (4,4), feat_dims3=(8,12), p=0.5):
         
        conv1 = torch.nn.Sequential(torch.nn.Conv2d(n_subj, out_ch1, (3,3)), \
                                torch.nn.BatchNorm2d(out_ch1),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        conv2 = torch.nn.Sequential(torch.nn.Conv2d(out_ch1, out_ch2, (3,3)), \
                                torch.nn.BatchNorm2d(out_ch2),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        conv3 = torch.nn.Sequential(torch.nn.Conv2d(out_ch2, out_ch3, (3,3)), \
                                torch.nn.BatchNorm2d(out_ch3),\
                                torch.nn.ReLU(),\
                                torch.nn.Dropout2d(p=p))
        decoder = torch.nn.Sequential(torch.nn.Conv2d(out_ch3, n_subj, (3,3)), \
                                torch.nn.BatchNorm2d(n_subj),\
                                torch.nn.Tanh())
        super(conv_autoencoder, self).__init__()
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
        self.decoder = decoder

    def forward(self, data):
        # encoder
        orig_dims = data.shape[-2:]
        batch_size = data.shape[0]
        out1 = self.conv1(data)
        out1 = F.interpolate(out1, self.feat_dims1)
        out2 = self.conv2(out1)
        out2 = F.interpolate(out2, feat_dims2)
        out3 = self.conv3(out2)
        enc_out = F.interpolate(out3, self.feat_dims3)
        # decoder
        dec_out = self.decoder(enc_out)
        dec_out = F.interpolate(dec_out, orig_dims)
        return enc_out, dec_out

def project_eeg(dataloader, model):
    '''Project EEG into new space using DNN model.
    Inputs:
        dataloader - dataloader which yields batches of size
                     (ims, subj, eeg_ch, eeg_time)
        model - DNN which returns encoder and out and accepts
                input of shape (ims, subj, eeg_ch, eeg_time)
    Ouputs:
        proj_eeg - tensor of shape (subj, ims, ch, time)
    '''
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # create datasets
    # EEG datasets
    datasets_dir = Path('/scratch/akitaitsev/intersubject_generalization/linear/',\
    'dataset_matrices/50hz/time_window13-40/')
    data_train = joblib.load(datasets_dir.joinpath('dataset_train.pkl'))
    data_test = joblib.load(datasets_dir.joinpath('dataset_test.pkl'))

    dataset_train = dataset(data_train)
    dataset_test = dataset(data_test)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,\
                                                    shuffle=True, num_workers=n_workers,\
                                                    drop_last=False)

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,\
                                                    shuffle=False, num_workers=n_workers,\
                                                    drop_last=False)

    # logging
    writer = SummaryWriter(out_dir.joinpath('runs'))

    # define the model
    model = conv_autoencoder(n_subj = data_test.shape[0])
    if gpu and n_workers >=1:
        warnings.warn('Using GPU and n_workers>=1 can cause some difficulties.')
    if gpu:
        device_name="cuda"
        device=torch.device("cuda:0")
        model.to(device)
        model=torch.nn.DataParallel(model)
        print("Using "+str(torch.cuda.device_count()) +" GPUs.")
    elif not gpu:
        device=torch.device("cpu")
        device_name="cpu"
        print("Using CPU.")

    # define the loss
    loss_fn = torch.nn.MSELoss() 

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

    # Loop through EEG dataset in batches
    for epoch in range(n_epochs):
        model.train()
        cntr=0
        tic = time.time()
        losses["epoch"+str(epoch)]=[]
        accuracies["epoch"+str(epoch)]=[]

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
        if cntr_epta % epta == 0:
            tic = time.time()
            
            # Project train and test set EEG into new space

            # treat encoder output as EEG
            eeg_train_projected_ENC = project_eeg(model, train_dataloader_for_assessment, layer="encoder")#, split_size=25) 
            eeg_test_projected_ENC = project_eeg(model, test_dataloader, layer = "encoder")  

            # treat projection head output as EEG
            eeg_train_projected_PH = project_eeg(model, train_dataloader_for_assessment)#, split_size=25) 
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
        cntr_epta += 1
    writer.close()

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    # Project EEG into new space using trained model
    projected_eeg = defaultdict()
    projected_eeg["train"] = defaultdict()
    projected_eeg["test"] = defaultdict() 
    projected_eeg["train"]["encoder"] = project_eeg(model, train_dataloader_for_assessment, layer="encoder", split_size=25) 
    projected_eeg["train"]["projection_head"] = project_eeg(model, train_dataloader_for_assessment, layer="encoder", split_size=25) 
    projected_eeg["test"]["encoder"] = project_eeg(model, test_dataloader, layer="proj_head") 
    projected_eeg["test"]["projection_head"] = project_eeg(model, test_dataloader, layer="proj_head") 
    
    # save projected EEG 
    joblib.dump(projected_eeg, out_dir.joinpath('projected_eeg.pkl'))

    # save trained model
    torch.save(model.state_dict(), out_dir.joinpath('trained_model.pt'))

    # save loss profile
    joblib.dump(losses, out_dir.joinpath('losses.pkl')) 
    
    # save test accuracy profile
    joblib.dump(accuracies, out_dir.joinpath('test_accuracies.pkl'))
