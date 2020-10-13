#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import librosa
import csv
import os
import torch
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
from utils import DynamicChunkSplitData
from utils import load_model


def Mel_Spec(fpath):
    signal, rate  = librosa.load(fpath, sr=16000)
    signal = signal/np.max(abs(signal)) # Restrict value between [-1,1]        
    mel_spec = librosa.feature.melspectrogram(signal, sr=16000, n_fft=512, hop_length=256, n_mels=128)
    mel_spec = mel_spec.T   
    return mel_spec

def get_audio_paths(inp_folder_path):
    # Walk the tree.
    pred_paths_all = []
    for root, directories, files in os.walk(inp_folder_path):
        files = sorted(files)
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            if '.wav' in filepath:
                pred_paths_all.append(filepath)
    return pred_paths_all
###############################################################################



# Parameters (desired input folder path)
input_path = './test/'

# loading normalize/de-normalize parameters
Feat_mean = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
Feat_std = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
Label_mean_Act = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
Label_std_Act = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
Label_mean_Dom = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
Label_std_Dom = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
Label_mean_Val = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
Label_std_Val = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]    

# loading audio paths for prediction process
paths = get_audio_paths(input_path)

# loading trained model
MODEL_PATH_ACT = './trained_models/Vgg16DeepEmoCluster_epoch50_batch64_10clusters_Act.pth.tar'
MODEL_PATH_DOM = './trained_models/Vgg16DeepEmoCluster_epoch50_batch64_30clusters_Dom.pth.tar'
MODEL_PATH_VAL = './trained_models/Vgg16DeepEmoCluster_epoch50_batch64_30clusters_Val.pth.tar'
model_act = load_model(MODEL_PATH_ACT, num_clusters=10)
model_dom = load_model(MODEL_PATH_DOM, num_clusters=30)
model_val = load_model(MODEL_PATH_VAL, num_clusters=30)
model_act.cuda()
model_act.eval()
model_dom.cuda()
model_dom.eval()
model_val.cuda()
model_val.eval()

# testing process
print('Start Prediction Process')
Pred_Act = []
Pred_Dom = []
Pred_Val = []
for i in tqdm(range(len(paths))):
    # extracting 128-Mel spec features
    data = Mel_Spec(paths[i])
    # Z-normalization
    data = (data-Feat_mean)/Feat_std
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    data[np.isnan(data)]=0
    data[data>3]=3
    data[data<-3]=-3         
    # chunk segmentation
    chunk_data = DynamicChunkSplitData([data], m=62, C=11, n=1)
    # numpy to GPU tensor & reshape input data to feed into model
    chunk_data = torch.from_numpy(chunk_data)
    chunk_data = chunk_data.view(chunk_data.size(0), 1, chunk_data.size(1), chunk_data.size(2))
    chunk_data = chunk_data.cuda()
    chunk_data = chunk_data.float()
    # models flow
    _, pred_act = model_act(chunk_data)
    _, pred_dom = model_dom(chunk_data)
    _, pred_val = model_val(chunk_data)
    # output
    Pred_Act.append(torch.mean(pred_act).data.cpu().numpy())
    Pred_Dom.append(torch.mean(pred_dom).data.cpu().numpy())
    Pred_Val.append(torch.mean(pred_val).data.cpu().numpy())

# Regression Task => De-Normalize Target and Prediction
Pred_Act = np.array(Pred_Act)
Pred_Dom = np.array(Pred_Dom)
Pred_Val = np.array(Pred_Val)
Pred_Act = (Label_std_Act*Pred_Act)+Label_mean_Act
Pred_Dom = (Label_std_Dom*Pred_Dom)+Label_mean_Dom
Pred_Val = (Label_std_Val*Pred_Val)+Label_mean_Val

# Output Predict Reulst
f = open('./pred_result.csv','w')
w = csv.writer(f)    
w.writerow(('File_Name','Aro-Score','Dom-Score','Val-Score')) 
for i in range(len(paths)):
    fname = paths[i].split('/')[-1]
    act = Pred_Act[i]
    dom = Pred_Dom[i]
    val = Pred_Val[i]
    w.writerow((fname, act, dom, val))         
f.close()           

