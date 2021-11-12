#! /usr/bin/bash

import torch
import numpy as np
import joblib
from pathlib import Path
from decord import VideoReader
from decord import cpu, gpu 
from PIL import Image
from torchvision import transforms as trn
from collections import defaultdict


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
    indices = np.linspace(0,total_frames-1,num_frames,dtype=np.int)
    for seg_ind in indices:
        images.append(Image.fromarray(vr[seg_ind].asnumpy()))
    return images, num_frames


def load_fmri(base_dir, track, region, dataset, subjs=None):
    '''Loads fMRI data as numpy arrays from Algonautus dataset.
    Returns a tuple of numpy arrays of fmri data and tuple of masks.
    Input:
        base_dir - str or Path object. Top Directory of the dataset.
        track - str, 'full_track' or 'mini_track'
        region - str, region for mini_track(EBA, FFA, LOC, PPA, STS, V1,
            V2, V3, V4, and WB for the whole brain data.
        dataset - str, train or test.
        subjs - list of str with numbers of subject (01, 02, etc.). Defualt=
            None, then using all 10 subjects.
    Outputs:
        fmri - tuple of numpy arrays of fMRI responses for the training set
        masks - tuple of numpy arrays of masks for respective fMRI responses for
            the training set
    '''
    if subjs==None:
        subjs=['01','02','03', '04', '05', '06', '07', '08', '09', '10']
    fmri=[]
    if dataset=='train':
        for subj in subjs:
            fl=joblib.load(Path(base_dir).joinpath("participants_data_v2021", track, \
                ('sub'+subj), (region+'.pkl')))
            fmri.append(fl["train"])
    elif dataset=='test':
        for subj in subjs:
            fl=joblib.load(Path(base_dir).joinpath("participants_data_v2021_test", track, \
                ('sub'+subj), ('organizers_data_'+region+'.pkl')))
            fmri.append(fl["test_data"])
    return fmri

def extract_features_cornet_s(video_list, model):
    '''
    Inputs:
        ims -list of PIL images sampled from one video of shape (each of shape (height,width, chs))
        lists of str:
        layers - layers of cornets, default=['V1', 'V2', 'V4', 'IT', 'decoder']
        sublayers - sublayers of cornets, default = ['output']
        timesteps - timesteps of cornets, default = [0, 1, 2, 3]
            Timesteps per layer in CORnet-S:
            # V1 --> 1 time step
            # V2 --> 2 time steps
            # V4 --> 4 time steps
            # IT --> 2 time steps
            # decoder --> 1 time step
    Outputs:
        stacked_acts
    '''
    def _extract_features_single_video(ims, model, layers=None, sublayers=None, timesteps=None):
        # resize normalize images
        resize_normalize = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # create an array of shape (frames, chs, height, width)
        ims = torch.stack([resize_normalize(im) for im in ims], dim=0)
        if torch.cuda.is_available():
            model.cuda()
            ims=ims.cuda().to(torch.float32)

        if layers is None:
            layers = ['V1', 'V2', 'V4', 'IT','decoder']
        if sublayers is None:
            sublayers = ['output']
        if timesteps is None:
            timesteps = [0, 1, 2, 3] 

        # hooker function to access the intermediate model features of particular 
        #layer and sublayer
        def _store_features(layer, inp, out):
            out = out.cpu().numpy()
            # (frames, features) for each time step
            _model_feats.append(np.reshape(out, (out.shape[0], -1))) 
            
        model_feats=[]
        activations=defaultdict()
        for layer in layers:
            for sublayer in sublayers:
                try:
                    m = model.module
                except:
                    m = model
                model_layer = getattr(getattr(m, layer), sublayer)
                model_layer.register_forward_hook(_store_features)
                _model_feats = []
                with torch.no_grad():
                    model(ims)
                for timestep in timesteps:
                    try:
                        model_feats.append(_model_feats[timestep])
                        print('Layer '+str(layer)+' sublayer '+str(sublayer)+' timestep '+str(timestep)\
                        +' '+str(_model_feats[timestep].shape))
                    except:
                        print('Layer has '+str(len(_model_feats))+' timesteps')
        return None            
    def _extract_features_multiple_videos(video_list, model, layers=None, sublayers=None, timesteps=None):
        for video in video_list: 
            _extract_features_single_video(video, model, layers=None, sublayers=None, timesteps=None)
        return None

    _extract_features_multiple_videos(video_list, model)
    return None


class feature_extractor(object):
    '''Feature extractor object.
    Applies .forward method of the model on every frame from video then averages the output of each layer over
    frames.
    Accepts custom functions for feature extraction (which are model-specific.

    .
    Parameters:
        model - class instance, DNN model to use for feature extraction. The .forward method of the
            model will be called on every frame (image) extracted from video. The default
            model is CORnet-S (https://github.com/dicarlolab/CORnet).

        feature_extraction_func - model-specific function to use for feature extraction.
            Default - extract_features_cornet_s
        '''
    def __init__(self, model, feature_extraction_function=None):
        self.model = model
        if feature_extraction_function is None:
            self.feature_extraction_function = extract_features_cornet_s
        else:
            self.feature_extraction_function = feature_extraction_function 

    def __call__(self, data):
        activations = self.feature_extraction_function(data, self.model)


if __name__ == '__main__':
    '''Run feature extraction with CORnet. Choose type of CORnet.
    Default- CORnet-S.
    '''
    import argparse
    import cornet
    import glob
    parser=argparse.ArgumentParser()
    parser.add_argument('-video_dir', type=str, help='Directory where videos are stored')
    parser.add_argument('-model', type=str, default='s', help='Type of cornet to use. s, r, or z.'
    'Default=s.')
    args = parser.parse_args()

    def get_cornet_model(pretrained=True, map_location=None):
        '''Get cornet model.'''
        model = model = getattr(cornet, f'cornet_{args.model.lower()}') 
        model = model(pretrained=pretrained, map_location=map_location)
        return model

    model = get_cornet_model()
    model.eval()
    video_list = glob.glob(args.video_dir + '/*.mp4')
    video_list.sort()

    feature_extractor = feature_extractor(model)
    videos=[]
#    for video in video_list:
#        vid, frames=sample_video_from_mp4(video)
#        videos.append(vid)
    video, frames=sample_video_from_mp4(video_list[0])
    videos.append(video)
    feature_extractor(videos)



