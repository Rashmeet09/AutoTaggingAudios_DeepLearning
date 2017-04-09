import numpy as np
import librosa               #for audio analysis
import matplotlib.pyplot as plt

def compute_melgram(audio_path):

    #Set Parameters
    SR = 12000               #Sampling rate i.e. no of samples per second of audio (in Hz)
    N_FFT = 512              #no of points in FFT i.e. FFT window length (in samples)#FFT = Fast Fourier Transform
    HOP_LEN = 256            #Hop length i.e. no of samples overlapping between 2 frames or windows of the signal (in samples)
    N_MELS = 96              #no of Mel bins
    DURA = 29.12             #(in seconds) #to make 1366 time-frame 

    '''
      N = total no of samples = DURA*SR
      T = total no of frames
      N = N_FFT + (T-1)*HOP_LEN 

      Output shape: (N_MELS, T)
    '''

    src, sr = librosa.load(audio_path, sr=SR)                                #load audio signal #requires audioread
    n_sample = src.shape[0]       
    n_sample_fit = int(DURA*SR)                                              #total 349440 samples

    if n_sample < n_sample_fit:                                              # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:                                            # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]

    melgram = librosa.feature.melspectrogram                                 #feature extraction
    logam = librosa.logamplitude
      
    S = melgram(y=src, sr=SR, hop_length=HOP_LEN,n_fft=N_FFT, n_mels=N_MELS) #S=mel-scaled Spectrogram
    ret = logam(S**2,ref_power=np.max)                                       #convert to log scale (dB)                                         
    print ret.shape                                                          #a mel-scaled spectrogram of shape ( 96,1366)


    plt.figure(figsize=(12,4))

    librosa.display.specshow(ret,y_axis='mel',fmax=9000,x_axis='time')                 #Display the spectrogram on a mel scale 

    plt.title('Mel-Spectrogram')
    plt.colorbar(format='%+02.0f dB')                                        # draw a color bar
    plt.tight_layout()                                                       # Make the figure layout compact 

    plt.show()

    ret = ret[np.newaxis, np.newaxis, :]
    print ret.shape                                                          #a mel-scaled spectrogram of shape (1,1,96,1366)
    return ret                                                       
