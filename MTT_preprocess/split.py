
#Python code to convert audios in MTT dataset to melgrams and split them for training,validation and testing
#You need to have MagnaTagATune (MTT) dataset containing 16 folders and annotations.csv file

import matplotlib.pyplot as plt
import numpy as np
import h5py
import librosa
import os, sys
import time
import pandas as pd
from sklearn.cross_validation import train_test_split 

PATH_MTT = '.../split/mtt/'                      #Path of MTT dataset
path_csv = '.../split/annotations_40tags.csv'    #Path of annotations.csv
PATH_HDF = '.../split/output_hdf/'               #Path where hdf5 files will be created
n_label = 40

# parameters for mel-spectrogram
SR = 12000            #Sampling rate i.e. no of samples per second of audio (in Hz)
max_len = 29.12       #(in seconds) #to make 1366 time-frame 
n_mels = 96           #no of Mel bins 
n_fft = 512           #no of points in FFT i.e. FFT window length (in samples)#FFT = Fast Fourier Transform
n_hop = 256           #Hop length i.e. no of samples overlapping between 2 frames or windows of the signal (in samples)
len_raw = int(SR * max_len)
n_freq = n_fft/2 + 1
mel_shape = librosa.feature.melspectrogram(np.zeros(int(SR*max_len)), SR, n_fft=n_fft, hop_length=n_hop, n_mels=n_mels).shape
print "\nmel shape:",mel_shape
n_mel_fr = mel_shape[1]
df = pd.read_csv(path_csv,header=0)


def train_validate_test_split(df,train_percent=0.7,validate_percent=0.15,seed=None):
    np.random.seed(7)
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test

#print df.dtypes
print "data-frame shape:",df.shape
n_data_all = df.shape[0]                        #train,valid,test .h5 output files contain 2 datasets:- 
                                                #melgram(from mtt audios) and y(from annotation file)
train, valid, test = train_validate_test_split(df)
n_train = len(train)
n_valid = len(valid)
n_test = len(test)
print "train:",n_train,"valid:", n_valid,"test:",n_test

np.random.seed(7)                                   # for reproducibility
#train_shfl_idxs = np.arange(n_train)
train_shfl_idxs = np.random.permutation(n_train)
valid_shfl_idxs = np.random.permutation(n_valid)
test_shfl_idxs = np.random.permutation(n_test) 
np.save('shuffled_idxs.npy', [train_shfl_idxs, valid_shfl_idxs, test_shfl_idxs]) 

def create_dataset_for(f_hdf, ds_name, num_data):
    newtype = np.dtype([('mp3','S200')])
    if ds_name == 'melgram':
        return f_hdf.create_dataset('melgram', (num_data, n_mels, n_mel_fr), dtype='float32')#hdf5 dataset having shape,datatype is returned
    elif ds_name == 'y':
        return f_hdf.create_dataset('y', (num_data, n_label), dtype='int64')
    elif ds_name == 'mp3':
        return f_hdf.create_dataset('mp3', (num_data, 1), dtype=newtype)
    else:
        print 'ha? %s?' % ds_name

def row_to_melgram(row_idx, row, dataset):
    ''' row: row of dataframe of pandas, dataset: a dataset of hdf file '''
    #fname, fold = row[2], row[3]
    fname = row[2]
    src, sr = librosa.load(PATH_MTT + fname, sr=SR)
    n_sample = src.shape[0]
    n_sample_fit = int(max_len*SR)

    if n_sample < n_sample_fit:    # if too short
        src = np.hstack((src, np.zeros((int(max_len*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]

    melgram = librosa.feature.melspectrogram
    logam = librosa.logamplitude

    S = melgram(y=src, sr=SR, hop_length=n_hop,n_fft=n_fft, n_mels=n_mels) #S=mel-scaled Spectrogram
    ret = logam(S**2,ref_power=1.0)                                        #convert to log scale (dB)  
    dataset[row_idx, :min(n_mel_fr, ret.shape[1])] = ret[:, :n_mel_fr]


def row_to_y(row_idx, row, dataset):          #for tags
    var = 4   
    while (var <= 43):
        dataset[row_idx, var-4] = row[var]
        var = var+1

def row_to_mp3(row_idx, row, dataset):        #for audio names
    dataset[row_idx, 0] = row[2]
     

def row_to(ds_name, row_idx, row, dataset):   #for audios
    if ds_name == 'melgram':
        row_to_melgram(row_idx, row, dataset)
    elif ds_name == 'y':
        row_to_y(row_idx, row, dataset)
    elif ds_name == 'mp3':
        row_to_mp3(row_idx, row, dataset)
   
def set_to_hdf(hdf_filepath, df_subset, shfl_idxs):
    ''' df_subset: pandas data frame, shfl_idxs: numpy integer array, shuffled index '''
    assert len(df_subset) == len(shfl_idxs), 'data frame length != indices list'
    start_time = time.time()
    num_data = len(df_subset)
    if os.path.exists(hdf_filepath):
        mode = 'a'
    else:
        mode = 'w'
    with h5py.File(hdf_filepath, mode) as f_hdf:                     #f_hdf is file object
        dataset1 = create_dataset_for(f_hdf, 'melgram', num_data)    #hdf5 dataset is returned
        dataset2 = create_dataset_for(f_hdf, 'y', num_data) 
        dataset3 = create_dataset_for(f_hdf, 'mp3', num_data) 
	print "dataset shape:",dataset1.shape,"dataset dtype:",dataset1.dtype
        for row_idx, row in enumerate(df_subset.iloc[shfl_idxs].itertuples()):   #index, item
            row_to_melgram(row_idx, row, dataset1)
            row_to_y(row_idx, row, dataset2)
            row_to_mp3(row_idx, row, dataset3)
            if row_idx % 20 == 0:
                sys.stdout.write('\r%d/%d-th sample was written.' % (row_idx+1, num_data))
    print '\n--- Done: It took %d seconds, %s ---' % (int(time.time() - start_time), hdf_filepath.split('/')[-1])
          

set_to_hdf(PATH_HDF+'train.h5', train, train_shfl_idxs)
set_to_hdf(PATH_HDF+'valid.h5', valid, valid_shfl_idxs)
set_to_hdf(PATH_HDF+'test.h5', test, test_shfl_idxs)  

for fname in ['train.h5','valid.h5','test.h5']:                #for standardisation
    for dname in ['melgram']:
        with h5py.File(PATH_HDF + fname, 'a') as f:
            mean = np.mean(f[dname])
            std = np.std(f[dname])
            f[dname][:] = (f[dname][:] - mean)/(std + np.finfo(np.float32).eps)

