#! /usr/bin/bash

import torch
import numpy as np
import os
import joblib
from pathlib import Path
from decord import VideoReader
from decord import cpu, gpu 
from PIL import Image
from torchvision import transforms as trn
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torchvision import transforms as trn
import torchextractor as tx
from sklearn.decomposition import PCA, IncrementalPCA
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable as V


def sample_video_from_mp4(file, num_frames=16):
    """This function takes a mp4 video file as input and returns
    a list of uniformly sampled frames (PIL Image).
    Parameters
    ----------
    file : str
        path to mp4 video file
    num_frames : int
        how many frames to select using uniform frame sampling.
    Returns
    -------
    images: list of PIL Images
    num_frames: int, number of frames extracted
    """
    images = list()
    vr = VideoReader(file, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0,total_frames-1,num_frames,dtype=int)
    for seg_ind in indices:
        images.append(Image.fromarray(vr[seg_ind].asnumpy()))
    return images, num_frames

def get_activations_and_save_image_model(model, video_list, activations_dir):
    """This function generates Alexnet features and save them in a specified directory.
    Parameters
    ----------
    model :
    pytorch model : alexnet.
    video_list : list
    the list contains path to all videos.
    activations_dir : str
    save path for extracted features.
    """

    resize_normalize = trn.Compose([
    trn.Resize((224,224)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for video_file in tqdm(video_list):
        vid,num_frames = sample_video_from_mp4(video_file)
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        activations = {}
        for frame,img in enumerate(vid):
            input_img = V(resize_normalize(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img=input_img.cuda()
            model_output, img_feature = model(input_img)
            for layer_name, f in img_feature.items():
                if frame==0:
                    activations[layer_name] = f.data.cpu().numpy().ravel()
                else:
                    activations[layer_name] += f.data.cpu().numpy().ravel()
            for layer_name, f in img_feature.items():
                save_path = os.path.join(activations_dir, video_file_name+"_"+layer_name + ".pkl")
                avg_layer_activation = activations[layer_name]/float(num_frames)
                joblib.dump(avg_layer_activation, save_path)


def do_PCA_and_save(activations_dir, save_dir,layers, n_components=1000):
    """This function preprocesses Neural Network features using PCA and save the results
    in  a specified directory
    .
    Parameters
    ----------
    activations_dir : str
    save path for extracted features.
    save_dir : str
    save path for extracted PCA features.
    layers: list
    list of strings with layer names to perform pca
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #train=[]
    #test=[]
    for layer in tqdm(layers):
        activations_file_list = glob.glob(activations_dir +'/*'+layer+'.pkl')
        activations_file_list.sort()
        feature_dim = joblib.load(activations_file_list[0])
        x = np.zeros((len(activations_file_list),feature_dim.shape[0]))
        for i,activation_file in enumerate(activations_file_list):
            temp = joblib.load(activation_file)
            x[i,:] = temp
        x_train = x[:1000,:]
        x_test = x[1000:,:]

        start_time = time.time()
        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)
        ipca = PCA(n_components=n_components,random_state=seed)
        ipca.fit(x_train)

        x_train = ipca.transform(x_train)
        x_test = ipca.transform(x_test)
        #train.append(x_train)
        #test.append(x_test)
    #train=np.concatenate(train, axis=1)
    #test=np.concatenate(test, axis=1)
        train_save_path = os.path.join(save_dir, layer+"_train_"+'activations'+'.pkl')
        test_save_path = os.path.join(save_dir, layer+ "_test_"+'activations'+'.pkl')
        joblib.dump(x_train, train_save_path)
        joblib.dump(x_test, test_save_path)


if __name__ == '__main__':
    '''Run feature extraction with resnet-18.
    '''
    import argparse
    import glob
    import time
    from pathlib import Path
    from sklearn.decomposition import PCA
    from torchvision.models import resnet18
    parser=argparse.ArgumentParser()
    parser.add_argument('-video_dir', type=str, help='Directory where videos are stored')
    parser.add_argument('-act_dir', type=str, help='Output directory to store '
    'extracted DNN activations.')
    parser.add_argument('-pca_dir', type=str, help='Output directory to store '
    'extracted DNN activations downsampled with PCA.')
    parser.add_argument('-n_comp', type=int, default=1000, help='Number of Principal Components to retain in data.')
    args = parser.parse_args()
    seed=0

    video_list = glob.glob(args.video_dir + '/*.mp4')
    video_list.sort()

    tic=time.time()
    # run feature extraction
    layers = ('layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc')
    model = resnet18(pretrained=True)
    model = tx.Extractor(model, layers)
    if torch.cuda.is_available():
        model=model.cuda()

    act_dir=Path(args.act_dir)
    pca_dir=Path(args.pca_dir)
    if not act_dir.is_dir():
        act_dir.mkdir(parents=True)
    if not pca_dir.is_dir():
        pca_dir.mkdir(parents=True)

    # activations
    get_activations_and_save_image_model(model, video_list, args.act_dir)
    
    # PCA
    do_PCA_and_save(args.act_dir, args.pca_dir, layers, args.n_comp)

    toc = time.time() - tic
    print('Elapsed time: '+str(toc))
