import time
import librosa
import numpy as np
from keras import backend as K
from CNNmodel import MusicTaggerCNN
#from music_tagger_cnn import MusicTaggerCNN
#from music_tagger_crnn import MusicTaggerCRNN
import audio_processor as ap
import pdb
import h5py

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#read it back
dataset1 = h5py.File('data/train.h5', 'r')
train_x = dataset1.get('melgram')
train_y = dataset1.get('y')

dataset2 = h5py.File('data/valid.h5', 'r')
valid_x = dataset2.get('melgram')
valid_y = dataset2.get('y')

dataset3 = h5py.File('data/test.h5', 'r')
test_x = dataset3.get('melgram')
test_y = dataset3.get('y')

def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]


def librosa_exists():
    try:
        __import__('librosa')
    except ImportError:
        return False
    else:
        return True


def main(net):

    print('Running main() with network: %s and backend: %s' % (net, K._BACKEND))
    # setting
    '''audio_paths = ['data/bensound-cute.mp3',
                   'data/bensound-actionable.mp3',
                   'data/bensound-dubstep.mp3',
                   'data/bensound-thejazzpiano.mp3']
    melgram_paths = ['data/bensound-cute.npy',
                     'data/bensound-actionable.npy',
                     'data/bensound-dubstep.npy',
                     'data/bensound-thejazzpiano.npy']
    #Y = np.loadtxt("data/annotations_final.csv")'''

    tags = ['rock', 'pop', 'alternative', 'indie', 'electronic',
            'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
            'beautiful', 'metal', 'chillout', 'male vocalists',
            'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
            '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
            'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
            'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
            '70s', 'party', 'country', 'easy listening',
            'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
            'Progressive rock', '60s', 'rnb', 'indie pop',
            'sad', 'House', 'happy'] 

    # prepare data like this
    '''melgrams = np.zeros((0, 1, 96, 1366))

    if librosa_exists:
        for audio_path in audio_paths:
            melgram = ap.compute_melgram(audio_path)
            melgrams = np.concatenate((melgrams, melgram), axis=0)
    else:
        for melgram_path in melgram_paths:
            melgram = np.load(melgram_path)
            melgrams = np.concatenate((melgrams, melgram), axis=0)
    #print "melgrams computed from audios" 
    '''

    # load model like this
    if net == 'cnn':
        model = MusicTaggerCNN(weights=None)

    print "Expected input shape:",model.input_shape
    #print melgrams.shape
    print "Actual input shape:",train_x.shape

    
    print "compiling"
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print "training"
    model.fit(train_x, train_y, nb_epoch=1, batch_size=20, shuffle=False)
    print "trained"
    scores = model.evaluate(valid_x, valid_y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
    # predict the tags like this
    print('Predicting...')
    start = time.time()
    pred_tags = model.predict(test_x)
    # print like this...
    print "Prediction is done. It took %d seconds." % (time.time()-start)
    print('Printing top-10 tags for each track...')
    for song_idx, audio_path in enumerate(test_x):
        sorted_result = sort_result(tags, pred_tags[song_idx, :].tolist())
        print(audio_path)
        print(sorted_result[:5])
        print(sorted_result[5:10])
        print(' ')
    
    

if __name__ == '__main__':

    networks = ['cnn']
    for net in networks:
    	main(net)
