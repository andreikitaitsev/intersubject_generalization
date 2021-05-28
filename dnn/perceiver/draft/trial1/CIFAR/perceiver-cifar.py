import torch
import torchvision
import torchvision.transforms as transforms
import perceiver as p
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-out_dir',type=str, default=\
'/scratch/akitaitsev/intersubject_generalizeation/dnn/perceiver/trial/')
parser.add_argument('-n_workers',type=int, default=1)
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-gpu',action='store_true', default=False)
parser.add_argument('-batch_per_loss',type=int, default=1)
args=parser.parse_args()


batch_size = args.batch_size
n_workers=args.n_workers
out_dir = Path(args.out_dir)
bpl = args.batch_per_loss


# logging
writer = SummaryWriter(out_dir.joinpath('runs'))    


transform = transforms.Compose(
    [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=n_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
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

if args.gpu:
    criterion = torch.nn.CrossEntropyLoss().cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

from collections import defaultdict
losses=defaultdict()
for epoch in range(2):  # loop over the dataset multiple times
    losses["epoch"+str(epoch)]=[]
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if args.gpu:
            inputs=inputs.cuda()
            labels=labels.cuda()
        # transpose inputs from (ims, chans, pix, pix) 
        # to (ims, pix, pix, RGB_chans)    
        inputs = torch.swapaxes(inputs, 1,3)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        losses["epoch"+str(epoch)].append(loss.cpu().detach().numpy())

        loss.backward()
        optimizer.step()

        # print statistics
        if i % bpl == 0:    
            # logging
            writer.add_scalar('training_loss', sum(losses["epoch"+str(epoch)][-bpl:])/bpl, \
                i+len(trainloader)*epoch)

            print('Epoch {:d}. Loss over iterations {:d}-{:d}: {:3f}.'.format(\
                epoch, i-bpl, i,  sum(losses["epoch"+str(epoch)][-bpl:])/bpl))
writer.close()

# save the model
torch.save(net.state_dict(), out_dir.joinpath('model.pt'))

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
