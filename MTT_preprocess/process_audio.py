#Run to compute and display spectrograms from audios

import time
import librosa
import numpy as np
import pdb
import h5py
import compute_melgram as ap


def librosa_exists():
    try:
        __import__('librosa')
    except ImportError:
        return False
    else:
        return True


def main():

    audio_paths = ['test-clip1.mp3','test-clip2.mp3']

    melgrams = np.zeros((0, 1, 96, 1366))

    if librosa_exists:
        for audio_path in audio_paths:
            melgram = ap.compute_melgram(audio_path)
            melgrams = np.concatenate((melgrams, melgram), axis=0)
    else:
        print "Install librosa"
    
    print "\nMelgrams computed from audios!\n" 
    

if __name__ == '__main__':

    main()
