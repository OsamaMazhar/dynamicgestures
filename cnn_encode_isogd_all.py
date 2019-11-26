# LSTM for sequence classification in the IMDB dataset
import numpy as np
import h5py
import time

from keras.layers import Input, Dense, LSTM, GRU, TimeDistributed, Flatten, Dropout, BatchNormalization, Conv1D
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential, clone_model
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, Adagrad
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.utils.vis_utils import plot_model
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
import os
import argparse
import sys

from rnn_utils import *

from keras.utils import multi_gpu_model
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

# fix random seed for reproducibility
np.random.seed(7)
non_zero_count = 0
zero_count = 0

set_name = 'train'

root_folder = '/media/osama/My-book/IsoGD/IsoGD_phase_1/'+set_name+'_ready/'
name_of_dataset = "full/top249_gestures_normed.h5"

# read_h5 = h5py.File("./gesture_5_dataset.h5", 'r')
read_h5 = h5py.File(root_folder+name_of_dataset, 'r')
print("Reading all_X_normed")
all_X_normed = read_h5['all_X_normed'][:]
print("Size of real_ges_vals is: ", np.shape(all_X_normed))
print("Reading y_one_hot.")
y_one_hot = read_h5['y_one_hot'][:]
print("Size of y_one_hot is: ", np.shape(y_one_hot))
real_ges_vals = read_h5['real_ges_vals'][:]
print("Size of real_ges_vals is: ", np.shape(real_ges_vals))
read_h5.close()

samples, time_steps, feature_set = np.shape(all_X_normed)
number_of_gestures = np.size(real_ges_vals)

print("Building model... - 1")
model_root_folder = '../../CNN/Trained_Models/InceptionV3-20180516105950/'
model_location = model_root_folder + 'e-745-vl-0.036-va-0.991.h5'
cnn_my = load_model(model_location)

for layer in cnn_my.layers:
    layer.trainable = False
print("Building model... - 2")
cnn_encode_output = (cnn_my.layers[-2].output)
cnn_encode = Model(inputs=cnn_my.input, outputs=[cnn_encode_output])

# print(cnn_encode.summary())

# >>> Load data now:
X_lhvn = np.memmap(root_folder+'full/left.dat',  dtype='float32', mode='r', shape=(samples, time_steps, 224, 224, 3))
X_rhvn = np.memmap(root_folder+'full/right.dat', dtype='float32', mode='r', shape=(samples, time_steps, 224, 224, 3))

# temp_output = cnn_encode(X_lhvn_Train[0,0,:,:,:])
temp = 1

left_image_embeddings  = np.zeros((samples, time_steps, 1024))
right_image_embeddings = np.zeros((samples, time_steps, 1024))

image_count = 0
for i in range(0, samples):
# for i in range(0, temp):
    for j in range(0, time_steps):
        left_image_input = np.expand_dims(X_lhvn[i,j,:,:,:], axis=0)
        right_image_input = np.expand_dims(X_rhvn[i,j,:,:,:], axis=0)
        left_image_embeddings[i,j,:] = cnn_encode.predict(left_image_input)
        right_image_embeddings[i,j,:] = cnn_encode.predict(right_image_input)
        image_count = image_count + 1
        print(str("Samples " + str(i+1) + " out of " + str(samples) + " and train images " + str(image_count) + " converted out of " + str(samples*time_steps) + "."), end ="\r", flush=True)
print("")

print("Writing the created vectors...")
file_h5 = h5py.File(root_folder+'full/image_embeddings_top_'+str(number_of_gestures)+'.h5', 'w')
file_h5.create_dataset('left_image_embeddings', data = left_image_embeddings)
# file_h5.create_dataset('right_image_embeddings', data = right_image_embeddings)
# # >>> Mistakenly gave name 'right_image_input' to 'right_image_embeddings'.
file_h5.create_dataset('right_image_input', data = right_image_embeddings)
# # <<< Mistakenly gave name 'right_image_input' to 'right_image_embeddings'.
file_h5.close()
print("Writing done...")
