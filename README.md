##Experiments on Automatic tagging of music using keras, tensorflow deep learning libraries and convolutional Neural Network (CNN), Convolutional Recurrent Neural Network (CRNN) deep learning architectures

##Environment Setup
You need to have the following enviornment setup on your laptop for Python Deep Learning with Keras:-
1. Python 2 or 3 installed and configured,
2. SciPy (including NumPy) installed and configured (e.g. via Anaconda),
3. Keras and a backend (Theano or TensorFlow) installed and configured

#Hardware specifications used
1. GPU - Geforce GTX 1080
2. 8 GB RAM

#Software specifications/Dependencies used
1. CUDA Toolkit - 8.0.44
2. NVIDIA Driver - 367.57
3. NVIDIA cuDNN - 5.1.5 (NVIDIA CUDA Deep Neural Network library)
4. GCC complier - 5.4.0
5. Python - 3.5.2 (pandas, numpy, matplotlib)
6. Tensorflow-gpu - 0.12.1
7. Keras - 1.2.1
8. Librosa - 0.4.3
9. Audioread - 2.1.4
10. h5py - 2.6.0

Refer the following links for installation:
1. http://www.pyimagesearch.com/2016/07/18/installing-keras-for-deep-learning/
2. https://charmie11.wordpress.com/2016/05/10/keras-deep-learning-library-installation-on-ubuntu-14-04/
3. http://ermaker.github.io/blog/2015/09/08/get-started-with-keras-for-beginners.html

##Dataset
 The magnatagatune dataset hosted by Music Informatics Research Group of City University London has been used to train the models. It consists of a collection of sound clips and corresponding human annotations collected by Edith Law's TagATune game.

 [Dataset link](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset): 16kHz 32kbps 25863 mp3 clips, each 29.1 s long

 1. Download the three mp3.zip files containing audio data, each 1 GB in size
 2. Merge the zip files
 'cat mp3.zip.001 mp3.zip.002 mp3.zip.003 < mp3.zip'
 3. Unzip the file
 'unzip mp3.zip'
 A dataset containing 16 folders from 0 to 9 and 'a' to 'f' is obatined.
 4. Download the tag annotations file in CSV format: 188 tags
 This file contains the names of the audio clips, their ids, the folder number and the binary values of tags, 1 indicating that the tag is present for the clip. A clip can have multiple tags, which makes this problem a mult-label classification problem.

##Selection of tags
The frequency of tags for each clip is plotted against the name of the clips. Some of the synonym tags are merged. You can find the different plots [here](CSV_process).
*Tags used (40)*
'''guitar, classical, slow, techno, string, vocal, electric, drum, no vocal, rock, fast, male, beat, female, piano, ambient, viloin, synthesizer, indian, singer, opera, quiet, harpsichord, loud, soft, flute, pop, sitar, choir, solo, new age, dance, weird, harp, heavy metal, cello, jazz, country, eastern, bazz'''

##Data Pre-processing
Splitting the dataset and processing the suidos by computing their melgrams is done.
Find the code [here](MTT_preprocess)

##Training
Deep learning architectures implemented :
1. Convolutional Neural Network(CNN)
2. Convolutional Recurrent Neural Network (CRNN)

##Results
The jupyter notebooks : [Results](Reports_ipynb)

##License
[MIT](LICENSE)
