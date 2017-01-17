import librosa
import numpy as np
import matplotlib.pyplot as plt

def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame
    parameters
    ----------
    audio_path: path for the audio file.
    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]

    melgram = librosa.feature.melspectrogram
    logam = librosa.logamplitude
   
    
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS),
                ref_power=np.max)
   

      # Make a new figure
    plt.figure(figsize=(12,4))

     # Display the spectrogram on a mel scale
     # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(ret)
     # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

     # draw a color bar
    plt.colorbar(format='%+02.0f dB')

     # Make the figure layout compact
    plt.tight_layout()
    #plt.show()
    ret = ret[np.newaxis, np.newaxis, :]
    #print "a"
    
    return ret
