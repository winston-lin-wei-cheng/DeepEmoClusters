#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import os
import librosa
import numpy as np
from scipy.io import savemat



"""
Fexed-Specs of Different Features (sampling rate=16000, mono channel)
1. mel-spec-feature : 512 window size, 256 step size [32ms with 16ms (50%) overlap], 128-mels
"""


def Mel_Spec(fpath):
    signal, rate  = librosa.load(fpath, sr=16000)
    signal = signal/np.max(abs(signal)) # Restrict value between [-1,1]        
    mel_spec = librosa.feature.melspectrogram(signal, sr=16000, n_fft=512, hop_length=256, n_mels=128)
    mel_spec = mel_spec.T   
    return mel_spec

def Extract_AcousticFeat(input_path, output_path):
    """
    Extract Mel-Spec Raw Acoustic Features
    Args:
        input_path$  (string): input directory to the training audio folder
        output_path$ (string): target directory to the acoustic feature folder
    """
    ERROR_record = ''
    # Walk the tree.
    for root, directories, files in os.walk(input_path):
        files = sorted(files)
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            if '.wav' in filepath:
                try:
                    features = Mel_Spec(filepath)
                    filename = filename.replace('wav','mat')
                    savemat(os.path.join(output_path, filename), {'Audio_data':features})
                except:
                    ERROR_record += 'Error: '+filename+'\n'                
            else:
                raise ValueError("Unsupport File Type!!")
    record_file = open("ErrorRecord.txt","w") 
    record_file.write(ERROR_record)
    record_file.close()
###############################################################################



if __name__=='__main__': 
      
    # Setting I/O Paths    
    input_path = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Audio/'
    output_path = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/Mel_Spec/feat_mat/'

    # creating output folder
    if not os.path.isdir(output_path):
        os.makedirs(output_path)  
   
    # Feature Extrators
    Extract_AcousticFeat(input_path, output_path)
