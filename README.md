# DeepEmoClusters
This is an implementation of semi-supervised [DeepEmoClusters]() framework for the attribute-based speech emotion recognition (SER) tasks. Part of the codes are contributed from the [DeepCluster](https://github.com/facebookresearch/deepcluster) repository. The experiments and trained models were based on the MSP-Podcast v1.6 corpus in the paper.


# Suggested Environment and Requirements
1. Python 3.6
2. Ubuntu 18.04
3. CUDA 10.0
4. pytorch version 1.4.0
5. librosa version 0.7.0
6. faiss version 1.6.0
7. The scipy, numpy and pandas packages
8. The MSP-Podcast corpus (request to download from [UTD-MSP lab website](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html))

# Feature Extraction & Preparation
Using the **feat_extract.py** to extract 128-mel spectrogram features for every speech segment in the corpus (remember to change I/O paths in the .py file). Then, use the **norm_para.py** to save normalization parameters for our framework's pre-processing step. The parameters will be saved in the generated *'NormTerm'* folder.

# How to run
After extracted the 128-mel spec features (e.g., Mel_Spec/feat_mat/\*.mat) for MSP-Podcast corpus, we use the *'labels_concensus.csv'* provided by the corpus as the default input label setting for the supervised emotional regressor network. 
1. change the size of the unlabeled set by the function *'getPaths_unlabel'* in **utils.py**. The default size of unlabeled set is 40K sentences.
2. change data & label paths in **main.py** for VGG-16 CNN model, the running args are,
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotional attributes (Act, Dom or Val)
   * -nc: number of clusters in the latent space for self-supervised learning
   * run in the terminal
   * the trained models will be saved under the generated *'Models'* folder
```
python main.py -ep 50 -batch 64 -emo Act -nc 10
```
3. change data & label & model paths & model parameters in **online_testing.py** for the evaluation results based on the MSP-Podcast test set,
   * run in the terminal
```
python online_testing.py -ep 50 -batch 64 -emo Act -nc 10
```

# Pre-trained models
We provide some trained models based on **version 1.6** of the MSP-Podcast in the *'trained_models'* folder. The CCC performances of models based on the test set are shown in the following table. Note that the results are slightly different from the [paper]() since we performed statistical test in the paper (i.e., we averaged multiple trails results together).

| 40K unlabeled set | Act(10-clusters) | Dom(30-clusters) | Val(30-clusters) |
|:----------------:|:----------------:|:----------------:|:----------------:|
| DeepEmoClusters | 0.6732 | 0.5547 | 0.1902 |


Users can get these results by running the **online_testing.py** with corresponding args.

# End-to-End Emotional Prediction Process
Since the framework is an end-to-end model, we also provide the complete prediction process that alows users to directly make emotional predictions (i.e., arousal, domiance and valence) for your own dataset or any audio files (audio spec: WAV file, 16k sampling rate and mono channel) based on the trained models from DeepEmoClusters. Users just need to change the input folder path in **prediction_process.py** to run the predictions and the output results will be saved as a *'pred_result.csv'* file under the same directory. 

# Reference
If you use this code, please cite the following paper:

Wei-Cheng Lin, Kusha Sridhar and Carlos Busso, "DeepEmoClusters: A Semi-Supervised Framework for Latent Cluster Representation of Speech Emotions"

```
@InProceedings{xXx,
  title={XXX},
  author={XXX},
  booktitle={XXX},
  year={2021},
} 
```
