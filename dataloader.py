#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winstonlin
"""
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import getPaths_attri, getPaths_unlabel
import random

# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)


class MspPodcastDataset(Dataset):
    """MSP-Podcast dataset (Unlabel only)"""

    def __init__(self, unlabel_dir):
        # init parameters
        self.unlabel_dir = unlabel_dir

        # label data paths
        self._paths = getPaths_unlabel(unlabel_dir)

        # norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']           

        # each utterance split into C chunks
        C = 11        
        self.imgs = []
        # every sentence becomes C chunks => repeat the same path for C times
        repeat_paths = self._paths.tolist()
        for i in range(len(repeat_paths)):
            self.imgs.extend([unlabel_dir+repeat_paths[i]]*C)
        
    def __len__(self):
        return len(getPaths_unlabel(self.unlabel_dir))
    
    def __getitem__(self, idx):
        # Loading Data
        data = loadmat(self.unlabel_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data'] 
        # Z-normalization
        data = (data-self.Feat_mean)/self.Feat_std
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        return data   

class MspPodcastEmoDataset(Dataset):
    """MSP-Podcast dataset (Label only)"""

    def __init__(self, root_dir, label_dir, split_set, emo_attr):
        # init parameters
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.split_set = split_set
        self.emo_attr = emo_attr

        # label data paths
        self._paths, self._labels = getPaths_attri(label_dir, split_set, emo_attr)

        # norm-parameters
        self.Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']         
        if emo_attr == 'Act':
            self.Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Dom':    
            self.Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
        elif emo_attr == 'Val':
            self.Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]   
        
        # each utterance split into C chunks
        C = 11        
        self.imgs = []
        # every sentence becomes C chunks => repeat the same path/label for C times
        repeat_paths = self._paths.tolist()
        repeat_labels = ((self._labels-self.Label_mean)/self.Label_std).tolist()
        for i in range(len(repeat_paths)):
            self.imgs.extend([(root_dir+repeat_paths[i], repeat_labels[i])]*C)
  
    def __len__(self):
        return len(getPaths_attri(self.label_dir, self.split_set, self.emo_attr)[0])
    
    def __getitem__(self, idx):
        # Loading Data
        data = loadmat(self.root_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data'] 
        # Z-normalization
        data = (data-self.Feat_mean)/self.Feat_std
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        # Loading Label & Normalization
        label = self._labels[idx]
        label = (label-self.Label_mean)/self.Label_std
        return data, label   
