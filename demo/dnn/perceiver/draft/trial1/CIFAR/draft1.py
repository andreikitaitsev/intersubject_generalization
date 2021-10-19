import torch
import torchvision
import torchvision.transforms as transforms
import perceiver as p
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-out_dir',type=str, default="/scratch/akitaitsev/"\
"intersubject_generalizeation/dnn/perceiver/trial/contrastive_loss-leftthomas-no_softmax/")
parser.add_argument('-n_epochs', type=int, default=20)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-n_workers',type=int, default=0)
parser.add_argument('-learning_rate', type=float, default=0.01)
parser.add_argument('-temperature', type=float, default=0.5)
parser.add_argument('-batch_per_loss', type=int, default=1)
parser.add_argument('-gpu',action='store_true', default=False)
args=parser.parse_args()

batch_size = args.batch_size
n_workers=args.n_workers
out_dir = Path(args.out_dir)
bpl = args.batch_per_loss

def Contrastive_Loss(out1, out2, batch_size, temperature, normalize=True):
    '''Inputs:
        out1, out2 - outputs of dnn input to which were batches of images yielded
                     from contrastive_loss_dataset
        batch_size - int
        temperature -int
    '''
    minval=1e-7
    if normalize:
        out1=F.normalize(out1, dim=1) #along rows
        out2=F.normalize(out2, dim=1)
    concat_out =torch.cat([out1, out2], dim=0)
    sim_matrix = torch.exp(torch.mm(concat_out, concat_out.t().contiguous()).clamp(min=minval)/temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


class ContrastiveLoss(torch.nn.Module):
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
    def __init__(self, data, transformer):
        '''data - 4d tensor of images (images, channels, pix, pix)
        transformer - random transformer object which 
        will be called on every geiitem call'''
        self.data = data
        self.transformer = transformer
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        augm1 = self.transformer(self.data[idx,:,:,:])
        augm2 = self.transformer(self.data[idx,:,:,:])
        return (augm1, augm2)

class my_optimizer(object):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.lr = torch.tensor(lr)
        self.momentum = torch.tensor(momentum)
        self.parameters = parameters
        if torch.cuda.is_available():
            self.lr = self.lr.cuda()
            self.momentum = self.momentum.cuda()
    def zero_grad(self):
        for par in self.parameters:
            if par.grad != None:
                par.grad.zero_()

    def step(self):
        for par in self.parameters:
            with torch.no_grad():
                par -= par*self.lr

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

degrees=(-10,10)
rotation = torchvision.transforms.RandomRotation(degrees)

transforms_random = torchvision.transforms.Compose([
    rotation, colorjitter]) # crop, colorjitter

transforms_common=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]) 



# random augmentation of training images
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=\
                                        transforms_common)
train_images, train_labels = zip(*trainset)
train_images = torch.stack(train_images, axis=0)
trainset = dataset_contrastive_loss(train_images, transforms_random)

import matplotlib.pyplot as plt
def im_show(im1, im2, idx):
    fig, (ax1,ax2)=plt.subplots(2)
    ax1.imshow(im1[idx], (2,1,0))
    ax2.imshow(im2[idx], (2,1,0))
    return fig

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=n_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=\
                                       transforms_common)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=n_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

n_classes=len(classes)

net = p.Perceiver(  
    input_channels = 3,          # number of channels for each token of the input
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
    num_classes = n_classes,          # output number of classesi = dimensionality of mvica output with 200 PCs
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
    self_per_cross_attn = 2      # number of self attention blocks per cross attention
    )

    
# transfer to GPU if available
if args.gpu:
    if torch.cuda.is_available():
        device="cuda:0"
        print("Using GPU.")
else:
    device="cpu"
    print('GPU is unavailable, using CPU.')
net.to(torch.device(device))
loss_fn = ContrastiveLoss(batch_size, args.temperature, device) 

#optimizer = my_optimizer(net.parameters(), lr=args.learning_rate)
#optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)


#from collections import defaultdict
#losses=defaultdict()
#for epoch in range(args.n_epochs):  # loop over the dataset multiple times
#    losses["epoch"+str(epoch)]=[]
#    i=0
#    grads=defaultdict()
#    params=defaultdict()
#    for batch1, batch2 in trainloader:
#        # transpose inputs from (ims, chans, pix, pix) 
#        # to (ims, pix, pix, RGB_chans)    
#        batch1 = torch.swapaxes(batch1, 1,3)
#        batch2 = torch.swapaxes(batch2, 1,3)
#        if args.gpu:
#            batch1=batch1.cuda()
#            batch2=batch2.cuda()
#        out1 = net(batch1)
#        out2 = net(batch2)
#        if args.gpu:
#            out1 = out1.cuda()
#            out2 = out2.cuda()
#        
#        optimizer.zero_grad()
#
#        # forward + backward + optimize
#        loss = Contrastive_Loss(out1, out2, args.batch_size, args.temperature) 
#        loss.backward()
#        
#        # debugging
##        params["step"+str(i)]=[]
##        for par in net.parameters():
##            params["step"+str(i)].append(par.cpu().detach().numpy())
##        grads["step"+str(i)]=[]
##        for par in net.parameters():
##            grads["step"+str(i)].append(par.grad.cpu().detach().numpy())
#        
#        optimizer.step()
#        losses["epoch"+str(epoch)].append(loss.cpu().detach().numpy())
#
#        # print statistics
#        if i % bpl == 0 and i!=0:    
#            # logging
#            writer.add_scalar('training_loss', sum(losses["epoch"+str(epoch)][-bpl:])/bpl, \
#                i+len(trainloader)*epoch)
#
#            print('Epoch {:d}. Loss over iterations {:d}-{:d}: {:3f}.'.format(\
#                epoch, i-bpl, i,  sum(losses["epoch"+str(epoch)][-bpl:])/bpl))
#        i+=1
#
#writer.close()
#
## save the model
#torch.save(net.state_dict(), out_dir.joinpath('model.pt'))
#
##save grads and params
#joblib.dump(params, out_dir.joinpath('params.pkl'))
#joblib.dump(grads, out_dir.joinpath('grads.pkl'))
#
# laod state dict 
net.load_state_dict(torch.load(out_dir.joinpath('model.pt')))
#

# test network
dataiter = iter(testloader)
images, labels = dataiter.next()
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = torch.swapaxes(images, 1,3)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy=(100*correct)/total
        print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)
        joblib.dump(correct, out_dir.joinpath('accuracy.pkl'))
