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
from keras.regularizers import l2, l1
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

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def conv_lstm_4(left_hand_input, skeleton_input, right_hand_input):

    # global cnn_encode
    input_left_hand = Input(shape=left_hand_input)
    input_skeleton = Input(shape=skeleton_input)
    input_right_hand = Input(shape=right_hand_input)

    left_model = TimeDistributed(Dense(512, activation='relu'))(input_left_hand)
    left_model = TimeDistributed(Dropout(drop_out))(left_model)
    left_model = LSTM(units=512, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(left_model)
    left_model = TimeDistributed(Dropout(drop_out))(left_model)
    left_model = TimeDistributed(BatchNormalization())(left_model)

    right_model = TimeDistributed(Dense(512, activation='relu'))(input_right_hand)
    right_model = TimeDistributed(Dropout(drop_out))(right_model)
    right_model = LSTM(units=512, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(right_model)
    right_model = TimeDistributed(Dropout(drop_out))(right_model)
    right_model = TimeDistributed(BatchNormalization())(right_model)

    skeleton_model = TimeDistributed(Dense(256, activation='relu'))(input_skeleton)
    skeleton_model = TimeDistributed(Dropout(drop_out))(skeleton_model)
    skeleton_model = LSTM(units=512, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(skeleton_model)
    skeleton_model = TimeDistributed(Dropout(drop_out))(skeleton_model)
    skeleton_model = TimeDistributed(BatchNormalization())(skeleton_model)

    concat_img_and_pv = concatenate([left_model, skeleton_model, right_model])
    # concat_img_and_pv = left_model

    # full_model = TimeDistributed(Dense(256, activation='relu'))(concat_img_and_pv)
    full_model = TimeDistributed(Dense(1536, activation='relu'))(concat_img_and_pv)
    full_model = LSTM(units=1024, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(full_model)
    full_model = TimeDistributed(Dropout(drop_out))(full_model)
    full_model = TimeDistributed(BatchNormalization())(full_model)

    full_model = LSTM(units=1024, return_sequences = False, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(full_model)
    full_model = Dense(1024, activation = "relu")(full_model)
    full_model = Dropout(drop_out)(full_model)
    full_model = Dense(249, activation = "softmax")(full_model)

    full_model = Model(inputs=[input_left_hand, input_skeleton, input_right_hand], outputs=full_model)
    return full_model

def conv_lstm(left_hand_input, skeleton_input, right_hand_input):

    # global cnn_encode
    input_left_hand = Input(shape=left_hand_input)
    input_skeleton = Input(shape=skeleton_input)
    input_right_hand = Input(shape=right_hand_input)

    # out = num_of_classes
    # sequence encoding
    # sequence_rnn = TimeDistributed(Dense(256, activation='relu'))(input_skeleton)
    # sequence_rnn = TimeDistributed(Dense(128, activation='relu'))(input_skeleton)
    # sequence_rnn = TimeDistributed(Dropout(0.4))(sequence_rnn)
    # sequence_rnn = TimeDistributed(Dense(64, activation='relu'))(sequence_rnn)
    # full model

    skeleton_model = TimeDistributed(Dense(256, activation='relu'))(input_skeleton)
    skeleton_model = TimeDistributed(Dropout(drop_out))(skeleton_model)
    skeleton_model = TimeDistributed(BatchNormalization())(skeleton_model)

    concat_img_and_pv = concatenate([input_left_hand, skeleton_model, input_right_hand])

    # full_model = TimeDistributed(Dense(256, activation='relu'))(concat_img_and_pv)
    full_model = TimeDistributed(Dense(1024, activation='relu'))(concat_img_and_pv)
    # x = LSTM(num_lstm, return_sequences=True, W_regularizer=l2(0.001), recurrent_dropout=0.4)
    full_model = GRU(units=1024, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001))(full_model)
    full_model = TimeDistributed(Dropout(drop_out))(full_model)
    full_model = TimeDistributed(BatchNormalization())(full_model)
    full_model = TimeDistributed(Dropout(drop_out))(full_model)

    # x = LSTM(64,dropout=0,5, recurrent_dropout=0,3)(x)
    full_model = GRU(units=1024, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001))(full_model)
    full_model = TimeDistributed(Dropout(drop_out))(full_model)
    full_model = TimeDistributed(BatchNormalization())(full_model)
    full_model = TimeDistributed(Dropout(drop_out))(full_model)

    full_model = GRU(units=1024,return_sequences = False, recurrent_dropout=drop_out)(full_model)
    full_model = Dense(512, activation='relu')(full_model)
    full_model = Dropout(drop_out)(full_model)
    full_model = BatchNormalization()(full_model)
    full_model = Dense(249, activation = "softmax")(full_model)

    full_model = Model(inputs=[input_left_hand, input_skeleton, input_right_hand], outputs=full_model)
    return full_model

def conv_lstm_3(left_hand_input, skeleton_input, right_hand_input):

    # global cnn_encode
    input_left_hand = Input(shape=left_hand_input)
    input_skeleton = Input(shape=skeleton_input)
    input_right_hand = Input(shape=right_hand_input)

    left_model = TimeDistributed(Dense(256, activation='relu'))(input_left_hand)
    left_model = TimeDistributed(Dropout(drop_out))(left_model)
    left_model = LSTM(units=512, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(left_model)
    left_model = TimeDistributed(Dropout(drop_out))(left_model)
    left_model = TimeDistributed(BatchNormalization())(left_model)

    right_model = TimeDistributed(Dense(256, activation='relu'))(input_right_hand)
    right_model = TimeDistributed(Dropout(drop_out))(right_model)
    right_model = LSTM(units=512, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(right_model)
    right_model = TimeDistributed(Dropout(drop_out))(right_model)
    right_model = TimeDistributed(BatchNormalization())(right_model)

    skeleton_model = TimeDistributed(Dense(256, activation='relu'))(input_skeleton)
    skeleton_model = TimeDistributed(Dropout(drop_out))(skeleton_model)
    skeleton_model = LSTM(units=256, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(skeleton_model)
    skeleton_model = TimeDistributed(Dropout(drop_out))(skeleton_model)
    skeleton_model = TimeDistributed(BatchNormalization())(skeleton_model)

    concat_img_and_pv = concatenate([left_model, skeleton_model, right_model])

    # full_model = TimeDistributed(Dense(256, activation='relu'))(concat_img_and_pv)
    full_model = TimeDistributed(Dense(1024, activation='relu'))(concat_img_and_pv)
    full_model = TimeDistributed(Dropout(drop_out))(full_model)
    full_model = TimeDistributed(BatchNormalization())(full_model)
    full_model = LSTM(units=512, return_sequences = True, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(full_model)
    full_model = TimeDistributed(Dropout(drop_out))(full_model)
    full_model = TimeDistributed(BatchNormalization())(full_model)

    full_model = LSTM(units=512, return_sequences = False, recurrent_dropout=drop_out, \
    bias_regularizer=l2(0.001), kernel_regularizer=l2(0.001), \
    recurrent_regularizer=l2(0.001))(full_model)
    full_model = Dense(256, activation = "relu")(full_model)
    full_model = Dropout(drop_out)(full_model)
    full_model = BatchNormalization()(full_model)
    full_model = Dense(249, activation = "softmax")(full_model)

    full_model = Model(inputs=[input_left_hand, input_skeleton, input_right_hand], outputs=full_model)
    return full_model

def read_skeleton_data(root_folder, name_of_dataset, *args, **kwargs):
    type = kwargs.get('type', None)
    if not type:
        type = ''
    read_h5 = h5py.File(root_folder+name_of_dataset, 'r')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Reading all_X_normed ' + type + '...')
    all_X_normed = read_h5['all_X_normed'][:]
    print('Reading y_one_hot ' +type+'...')
    y_one_hot = read_h5['y_one_hot'][:]
    print('Reading real_ges_vals ' +type+'...')
    real_ges_vals = read_h5['real_ges_vals'][:]
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Size of all_X_normed '+type + ' is:  ', np.shape(all_X_normed))
    print('Size of y_one_hot '+type+' is:     ', np.shape(y_one_hot))
    print('Size of real_ges_vals '+type+' is: ', np.shape(real_ges_vals))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    read_h5.close()
    return all_X_normed, y_one_hot, real_ges_vals

def read_image_embeddings(root_folder, name_of_image_embds, *args, **kwargs):
    type = kwargs.get('type', None)
    if not type:
        type = ''
    read_h5 = h5py.File(root_folder+name_of_image_embds, 'r')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Reading the left hand image '+type+' embeddings...')
    left_image_embeddings = read_h5['left_image_embeddings'][:]
    print('Reading the right hand image '+type+' embeddings...')
    right_image_embeddings = read_h5['right_image_input'][:]
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Size of left_image_embeddings '+type+' is:  ', np.shape(left_image_embeddings))
    print('Size of right_image_embeddings '+type+' is: ', np.shape(right_image_embeddings))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    read_h5.close()
    return left_image_embeddings, right_image_embeddings

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
K.set_session(sess)

np.random.seed(7)
drop_out = 0.7
non_zero_count = 0
zero_count = 0

train_root_folder = "/media/osama/My-book/IsoGD/IsoGD_phase_1/train_ready/full/"
valid_root_folder = "/media/osama/My-book/IsoGD/IsoGD_phase_1/valid_ready/full/"

name_of_dataset = "top249_gestures_normed.h5"
name_of_image_embds = "image_embeddings_top_249.h5"

all_X_normed_train, y_one_hot_train, real_ges_vals_train = \
read_skeleton_data(train_root_folder, name_of_dataset, type='train')

random_index_train = np.arange(np.shape(all_X_normed_train)[0])
portion_train = int(np.shape(all_X_normed_train)[0]/1)

np.random.shuffle(random_index_train)
random_index_train = random_index_train[:portion_train]

left_image_embeddings_train, right_image_embeddings_train = \
read_image_embeddings(train_root_folder, name_of_image_embds, type='train')

all_X_normed_train = all_X_normed_train[random_index_train]
left_image_embeddings_train = left_image_embeddings_train[random_index_train]
right_image_embeddings_train = right_image_embeddings_train[random_index_train]
y_one_hot_train = y_one_hot_train[random_index_train]

all_X_normed_valid, y_one_hot_valid, real_ges_vals_valid = \
read_skeleton_data(valid_root_folder, name_of_dataset, type='valid')

random_index_valid = np.arange(np.shape(all_X_normed_valid)[0])
portion_valid = int(np.shape(all_X_normed_valid)[0]/1)

np.random.shuffle(random_index_valid)
random_index_valid = random_index_valid[:portion_valid]

left_image_embeddings_valid, right_image_embeddings_valid = \
read_image_embeddings(valid_root_folder, name_of_image_embds, type='valid')

all_X_normed_valid = all_X_normed_valid[random_index_valid]
left_image_embeddings_valid = left_image_embeddings_valid[random_index_valid]
right_image_embeddings_valid = right_image_embeddings_valid[random_index_valid]
y_one_hot_valid = y_one_hot_valid[random_index_valid]

train_samples, time_steps, feature_set = np.shape(all_X_normed_train)
number_of_gestures = np.size(real_ges_vals_train)

print("Building model...")
model = conv_lstm(left_hand_input = (40, 1024), skeleton_input = (40, 129), right_hand_input = (40, 1024))
print("Model built...")
# plot_model(model, to_file='isogd_model.png', show_shapes=True, show_layer_names=True)
print(model.summary())
# opt = Adagrad(lr=0.01, epsilon=None, decay=0.0001)
opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0001)

NUM_GPU = 2
NUM_EPOCHS = 100000
BATCH_SIZE = 800

# >>> Load data now:
# X_lhvn_Train = np.memmap('./X_lhvn_Train.dat', dtype='float32', mode='r', shape=(train_samples, time_steps, 224, 224, 3))
# X_rhvn_Train = np.memmap('./X_rhvn_Train.dat', dtype='float32', mode='r', shape=(train_samples, time_steps, 224, 224, 3))

# >>> Saves the model weights after each epoch if the validation loss decreased
output_dir = '/media/osama/My-book/IsoGD/IsoGD_phase_1/ISOGesture249/Trained_Models/'
now = datetime.now()
nowstr = now.strftime('dgesture-%Y%m%d%H%M%S')
now = os.path.join(output_dir, nowstr)
# <<< Saves the model weights after each epoch if the validation loss decreased

os.makedirs( now, exist_ok=True )

# >>> Create our callbacks
savepath = os.path.join( now, 'e-{epoch:03d}-vl-{val_loss:.3f}-va-{val_acc:.3f}.h5' )
savepath_log = os.path.join( now, 'CSV_Log.csv' )

checkpointer = ModelCheckpoint(filepath=savepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# checkpointer = ModelCheckpoint(filepath=savepath, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
csv_logger = CSVLogger(savepath_log, append=True, separator=';')
# tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
callbacks_list = [checkpointer, csv_logger]
# <<< Create our callbacks

if NUM_GPU > 1:
    parallel_model = multi_gpu_model(model, NUM_GPU)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # parallel_model.fit([left_image_embeddings_train, xTrain, right_image_embeddings_train], y_one_hot_train, validation_split=val_split, shuffle=True, batch_size = BATCH_SIZE * NUM_GPU, epochs=NUM_EPOCHS, callbacks=callbacks_list)
    parallel_model.fit([left_image_embeddings_train, all_X_normed_train, right_image_embeddings_train], \
    y_one_hot_train, validation_data=([left_image_embeddings_valid, all_X_normed_valid, right_image_embeddings_valid], y_one_hot_valid), \
    shuffle=True, batch_size = BATCH_SIZE * NUM_GPU, epochs=NUM_EPOCHS, callbacks=callbacks_list)
else:
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.fit([left_image_embeddings_train, xTrain, right_image_embeddings_train], y_one_hot_train, validation_split=val_split, shuffle=True, batch_size = BATCH_SIZE * NUM_GPU, epochs=NUM_EPOCHS, callbacks=callbacks_list)
    model.fit([left_image_embeddings_train, all_X_normed_train, right_image_embeddings_train], \
    y_one_hot_train, validation_data=([left_image_embeddings_valid, all_X_normed_valid, right_image_embeddings_valid], y_one_hot_valid), \
    shuffle=True, batch_size = BATCH_SIZE * NUM_GPU, epochs=NUM_EPOCHS, callbacks=callbacks_list)

# loss, acc = parallel_model.evaluate([xTest, X_lhvn_Test, X_rhvn_Test], y_one_Test)
# print("Dev set acc score = ", acc)
# for i in range(np.shape(xTest)[0]):
#     test_input_index = i
#     test_input = np.expand_dims(xTest[test_input_index,:,:], axis=0)
#     predictions = model.predict(test_input)
#     predictions = (predictions > 0.6).astype(int)
#     print("-------------------------------------------------------------")
#     print("predicted: ", predictions[0])
#     print("original: ", (y_one_test[test_input_index,:].astype(int)))
#     print("-------------------------------------------------------------")
