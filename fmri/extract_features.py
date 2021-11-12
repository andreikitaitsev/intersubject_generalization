#! /usr/bin/bash

import torch
import numpy as np
import joblib
import tqdm
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

def extract_features_cornet_s(video_list, model, debug=False):
    '''
    Extract intermediate features of CORnet-S. By default, uses all the timesteps of 
    V1, V2, V4, IT, decoder layers, sublayer = output.

    Timesteps per layer in CORnet-S:
    # V1 --> 1 time step
    # V2 --> 2 time steps
    # V4 --> 4 time steps
    # IT --> 2 time steps
    # decoder --> 1 time step
    
    Inputs:
        video_list - list of lists of PIL images sampled from one videos by sample_video_from_mp4  
        model - pytorch CORnet-S model (https://github.com/dicarlolab/CORnet)
        debug - bool. If true, prints the shape the output of each layer, sublayer and timestep.
            Default=False
    Outputs:
        activations - 2d numpy array of shape (n_videos, concat_features), where concat_features
            are the concatenated outputs of all CORnet-S layers and timesteps averaged over frames 
            for each video.
    '''
    def _extract_features_single_video_single_layer(ims, model, layer):
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

        timestep_dict={'V1': [0], 'V2':[0,1], 'V4':[0,1,2,3],'IT':[0,1],'decoder':[0]}
        sublayer = 'output'

        # hooker function to access the intermediate model features of particular 
        #layer and sublayer
        def _store_features(layer, inp, out):
            out = out.cpu().numpy()
            # (frames, features) for each time step
            _model_feats.append(np.reshape(out, (out.shape[0], -1))) 
            
        model_feats=[]
        activations=defaultdict()
        try:
            m = model.module
        except:
            m = model
        model_layer = getattr(getattr(m, layer), sublayer)
        model_layer.register_forward_hook(_store_features)
        _model_feats = []
        with torch.no_grad():
            model(ims)
        timesteps=timestep_dict[str(layer)]
        for timestep in timesteps:
            try:
                model_feats.append(_model_feats[timestep])
                if debug:
                    print('Layer: '+str(layer)+' sublayer: '+str(sublayer)+' timestep: '+str(timestep)\
                    +'\n'+str(_model_feats[timestep].shape))
            except:
                if debug:
                    print('Attempted to access timestep '+str(timestep)+\
                        'but \n layer '+str(layer)+' has '+str(len(timesteps))+' timesteps.')
        # concatenate along features if a layer has multiple time steps. If layer has one time step, 
        # convert it to np array
        if len(model_feats) > 1:
            model_feats=np.concatenate(model_feats, axis=1)
        elif len(model_feats) == 1:
            model_feats=model_feats[0]
        return model_feats            

    def _extract_features_single_video_multiple_layers(ims, model, layers=None):
        '''Extract features from multiple layers for one video, averages features over frames for every 
        layer and concatenates features from different layers
        Inputs:
            ims - np array of shape (n_frames, features)
            model - torch model, CORnet-S
            layers - list of str, default = V1, V2, V4, IT, decoder
        Outputs:
            acts - 1d np.array of concatenated activations for every layer averaged over frames 
        '''
        if layers is None:
            layers = ('V1', 'V2', 'V4', 'IT', 'decoder')
        acts=[]
        for layer in layers:
            # average activations over frames for each layer
            acts.append(np.mean(_extract_features_single_video_single_layer(ims, model, layer), axis=0))
        acts = np.concatenate(acts)
        return acts

    def _extract_features_multiple_videos(video_list, model, layers=None):
        '''Extract features from every video and return array of shape (n_videos, concat_features)
        Inputs:
            video_list - list of lists of PIL images sampled from different videos
            model - torch model, CORnet-S
            layers - list of layers to use. Default - ['V1', 'V2', 'V4', 'IT', 'decoder']
        Outputs:
            activations - 2d numpy array of shape (n_videos, n_features), where n_features are features
                of different layers and timesteps of cornet-s averaged over frames for every video and 
                concatenated 
        '''
        activations=[]
        for video in tqdm(video_list): 
            activations.append(_extract_features_single_video_multiple_layers(video, model, layers))
        activations=np.stack(activations, axis=0)
        return activations

    activations = _extract_features_multiple_videos(video_list, model)
    return activations


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
        posprocessor - object, shall implement .fit_transform or .fit and .transform method.
            these methods will be called on the ouput of the feature extraction function.
            Assumed to be sklearn transformer or Pipeline object. Default=None-no posprocessing
    Methods:
        __call__ - applies feature_extraction function on the input data.
        '''
    def __init__(self, model, feature_extraction_function=extract_features_cornet_s,\
        postprocessor=None):
        self.feature_extraction_function = feature_extraction_function
        self.model = model
        self.postprocessor = postprocessor

    def __call__(self, data):
        activations = self.feature_extraction_function(data, self.model)
        if not self.postprocessor is None:
            # check if postprocessor has fit_transform or fit and transform methods   
            if getattr(self.postprocessor, 'fit_transform', False) and\
                callable(getattr(self.postprocessor, 'fit_transform'):
                activations = self.postprocessor.fit_transform(activations)
            elif getattr(self.postprocessor, 'fit', False) and getattr(self.postprocessor, 'transform', False)\
                and getattr(self.postprocessor, 'fit') and getattr(self.postprocessor, 'transform')):
                self.postprocessor.fit(activations)
                activations = self.postprocessor.transform(activations)
        return activations


if __name__ == '__main__':
    '''Run feature extraction with CORnet. Choose type of CORnet.
    Default- CORnet-S.
    '''
    import argparse
    import cornet
    import glob
    from sklearn.decomposition import PCA
    parser=argparse.ArgumentParser()
    parser.add_argument('-video_dir', type=str, help='Directory where videos are stored')
    parser.add_argument('-model', type=str, default='s', help='Type of cornet to use. s, r, or z.'
    'Default=s.')
    parser.add_argument('-n_comp', type=int, help='Number of Principal Components to retain in data.')
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

    # configure feature extractor
    postprocessor = PCA(args.n_comp)
    feature_extractor = feature_extractor(model, extract_features_cornet_s, postprocessor)

    videos=[]
    for video in video_list:
        vid, frames=sample_video_from_mp4(video)
        videos.append(vid)
    activations = feature_extractor(videos)
    print(activations.shape)

