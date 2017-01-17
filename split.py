#Python code to convert audios in MTT dataset to melgrams and split them for training,validation and testing
#You need to have MagnaTagATune (MTT) dataset containing 16 folders and annotations.csv file

import matplotlib.pyplot as plt
import numpy as np
import h5py
import librosa
import os, sys
import time
import pandas as pd 

PATH_MTT = '/home/rashmeet/Pictures/mtt/'
path_csv = '/home/rashmeet/Pictures/annotations_final_4.csv'
PATH_HDF = '/home/rashmeet/Pictures/output_hdf/'
n_label = 37

# audios
SR = 12000            #sampling rate in Hz
max_len = 29.12       # in Seconds
n_mels = 96           #no of mel-bins 
n_fft = 512           #length of fft window
n_hop = 256           #no of audio samples betn successive frames
len_raw = int(SR * max_len)
n_freq = n_fft/2 + 1

mel_shape = librosa.feature.melspectrogram(np.zeros(SR*max_len), SR, n_fft=n_fft, hop_length=n_hop, n_mels=n_mels).shape
print "\nmel shape:",mel_shape
n_mel_fr = mel_shape[1]
df = pd.read_csv(path_csv,header=0)

#print df.dtypes
print "data-frame shape:",df.shape
n_data_all = df.shape[0]                        #train,valid,test .h5 output files contain 2 datasets:- 
n_train = len(df[df['fold'].isin([6,7,8])])     #melgram(from mtt audios) and y(from annotation file)
#n_valid = len(df[df['fold'].isin([12,13])])
#n_test = len(df[df['fold'].isin([14,15])])
#n_train = n_data_all - n_valid - n_test
#print "train:",n_train,"valid:", n_valid,"test:", n_test

np.random.seed(1)  # for reproducibility
#train_shfl_idxs = np.arange(n_train)
train_shfl_idxs = np.random.permutation(n_train)
#valid_shfl_idxs = np.random.permutation(n_valid)
#test_shfl_idxs = np.random.permutation(n_test)
np.save('shuffled_idxs.npy', [train_shfl_idxs]) 
#np.save('shuffled_idxs.npy', [train_shfl_idxs, valid_shfl_idxs, test_shfl_idxs]) 

def create_dataset_for(f_hdf, ds_name, num_data):
    if ds_name == 'melgram':
        return f_hdf.create_dataset('melgram', (num_data, n_mels, n_mel_fr), dtype='float32')#hdf5 dataset having shape,datatype is returned
    elif ds_name == 'y':
        return f_hdf.create_dataset('y', (num_data, n_label), dtype='int64')
    else:
        print 'ha? %s?' % ds_name

def row_to_melgram(row_idx, row, dataset):
    ''' row: row of dataframe of pandas, dataset: a dataset of hdf file '''
    fname, fold = row[2], row[3]
    src, sr = librosa.load(PATH_MTT + fname, SR)
    melgram = librosa.feature.melspectrogram(src, sr, n_fft=n_fft,hop_length=n_hop, n_mels=n_mels)
    melgram = np.abs(melgram) ** 2
    dataset[row_idx, :, :min(n_mel_fr, melgram.shape[1])] = melgram[:, :n_mel_fr]

def row_to_y(row_idx, row, dataset):
    var = 4   
    while (var <= 40):
        dataset[row_idx, var-4] = row[var]
        var = var+1

def row_to(ds_name, row_idx, row, dataset):
    if ds_name == 'melgram':
        row_to_melgram(row_idx, row, dataset)
    elif ds_name == 'y':
        row_to_y(row_idx, row, dataset)
   
def set_to_hdf(hdf_filepath, df_subset, shfl_idxs):
    ''' df_subset: pandas data frame, shfl_idxs: numpy integer array, shuffled index '''
    assert len(df_subset) == len(shfl_idxs), 'data frame length != indices list'
    start_time = time.time()
    num_data = len(df_subset)
    if os.path.exists(hdf_filepath):
        mode = 'a'
    else:
        mode = 'w'
    with h5py.File(hdf_filepath, mode) as f_hdf:                  #f_hdf is file object
        dataset1 = create_dataset_for(f_hdf, 'melgram', num_data)    #hdf5 dataset is returned
        dataset2 = create_dataset_for(f_hdf, 'y', num_data) 
	print "dataset shape:",dataset1.shape,"dataset dtype:",dataset1.dtype
        for row_idx, row in enumerate(df_subset.iloc[shfl_idxs].itertuples()):   #index, item
            row_to_melgram(row_idx, row, dataset1)
            row_to_y(row_idx, row, dataset2)
            if row_idx % 20 == 0:
                sys.stdout.write('\r%d/%d-th sample was written.' % (row_idx+1, num_data))
    print '\n--- Done: It took %d seconds, %s ---' % (int(time.time() - start_time), hdf_filepath.split('/')[-1])
          

set_to_hdf(PATH_HDF+'train.h5', df[df['fold'].isin([6,7,8])], train_shfl_idxs)
#set_to_hdf(PATH_HDF+'valid.h5', df[df['fold'].isin([12,13])], valid_shfl_idxs)
#set_to_hdf(PATH_HDF+'test.h5', df[df['fold'].isin([14,15])], test_shfl_idxs)    

