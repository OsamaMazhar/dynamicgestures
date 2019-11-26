
import argparse
import numpy as np
import glob
from PIL import Image
import h5py

from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
from rnn_utils import *
import datetime

np.random.seed(7)

train_root_folder = "/media/osama/My-book/IsoGD/IsoGD_phase_1/train_ready/full/"
valid_root_folder = "/media/osama/My-book/IsoGD/IsoGD_phase_1/valid_ready/full/"
test_root_folder = "/media/osama/My-book/IsoGD/IsoGD_phase_1/test_ready/full/"

name_of_dataset = "top249_zp_gestures_together_normed.h5"
name_of_image_embds = "image_embeddings_top_249.h5"
name_of_stand = "top249_zp_gestures_standard_normed.h5"

y_one_hot_train, real_ges_vals_train = \
read_skeleton_data(train_root_folder, name_of_dataset, type='train')

all_X_standard_normed_train = read_standard_normed_pose(train_root_folder, name_of_stand, type='train')

left_image_embeddings_train, right_image_embeddings_train = \
read_image_embeddings(train_root_folder, name_of_image_embds, type='train')

y_one_hot_valid, real_ges_vals_valid = \
read_skeleton_data(valid_root_folder, name_of_dataset, type='valid')

all_X_standard_normed_valid = read_standard_normed_pose(valid_root_folder, name_of_stand, type='valid')

left_image_embeddings_valid, right_image_embeddings_valid = \
read_image_embeddings(valid_root_folder, name_of_image_embds, type='valid')

y_one_hot_test, real_ges_vals_test = \
read_skeleton_data(test_root_folder, name_of_dataset, type='test')

all_X_standard_normed_test = read_standard_normed_pose(test_root_folder, name_of_stand, type='test')

left_image_embeddings_test, right_image_embeddings_test = \
read_image_embeddings(test_root_folder, name_of_image_embds, type='test')

X_all = np.vstack((all_X_standard_normed_train, all_X_standard_normed_valid, all_X_standard_normed_test))
left_all = np.vstack((left_image_embeddings_train, left_image_embeddings_valid, left_image_embeddings_test))
right_all = np.vstack((right_image_embeddings_train, right_image_embeddings_valid, right_image_embeddings_test))
y_all = np.vstack((y_one_hot_train, y_one_hot_valid, y_one_hot_test))

random_index_train = np.arange(np.shape(X_all)[0])
np.random.shuffle(random_index_train)

X_all = X_all[random_index_train]
left_all = left_all[random_index_train]
right_all = right_all[random_index_train]
y_all = y_all[random_index_train]

# x_train = X_all[12000:]
# left_train = left_all[12000:]
# right_train = right_all[12000:]
# y_train = y_all[12000:]
#
# x_valid = X_all[:6000]
# left_valid = left_all[:6000]
# right_valid = right_all[:6000]
# y_valid = y_all[:6000]

x_test = X_all[6000:12000]
left_test = left_all[6000:12000]
right_test = right_all[6000:12000]
y_test = y_all[6000:12000]

# train_samples, time_steps, feature_set = np.shape(x_train)
# number_of_gestures = np.size(real_ges_vals_train)

print("Size of x_test: ", np.shape(x_test))
print("Size of y_test: ", np.shape(y_test))
print("Size of left_test: ", np.shape(left_test))
print("Size of right_test: ", np.shape(right_test))

model_location = '/media/osama/My-book/IsoGD/IsoGD_phase_1/ISOGesture249/Trained_Models/dgesture-20190812122154/e-658-vl-0.632-va-0.866.h5'
model = load_model(model_location)

num_of_gestures = np.size(real_ges_vals_train)

Q = np.zeros((num_of_gestures, num_of_gestures), dtype=int)
print("Shape of Q", Q.shape)

correct_preds = 0
wrong_preds = 0
time_vector = []
prediction_threshold = 0.65
for i in range(np.shape(x_test)[0]):
    test_input_index = i
    test_input = np.expand_dims(x_test[test_input_index,:,:], axis=0)
    left_input = np.expand_dims(left_test[test_input_index,:,:], axis=0)
    right_input = np.expand_dims(right_test[test_input_index,:,:], axis=0)
    start_time = datetime.datetime.now()
    predictions = model.predict([left_input, test_input, right_input])
    end_time = datetime.datetime.now()
    time_taken = end_time - start_time
    time_taken_milli = int(time_taken.total_seconds() * 1000)
    time_vector = np.append(time_vector, time_taken_milli)
    avg_time = np.mean(time_vector)
    orig_index = np.where(y_test[test_input_index,:] == 1)
    Q[orig_index[0][0], np.argmax(predictions).astype(int)] += 1
    predictions = (predictions > prediction_threshold).astype(int)
    # print("-------------------------------------------------------------")
    # # print("predicted: ", np.argmax(predictions).astype(int))
    # print("predicted: ", predictions[0])
    # print("original:  ", (y_test[test_input_index,:].astype(int)))
    # print("-------------------------------------------------------------")
    predicted = predictions[0]
    original = y_test[test_input_index,:].astype(int)
    if (predicted == original).all():
        correct_preds = correct_preds + 1
    else:
        wrong_preds = wrong_preds + 1
    print("True Predictions: " + str(correct_preds) + ", False Predictions: " + str(wrong_preds) + " in average time: " + str(avg_time) + " ms.   ", end ="\r", flush=True)
print("")
print("Results Quantified!!!")
score = model.evaluate([left_test, x_test, \
right_test], y_test)
print("Model Accuracy: ", score)

print("Q Matrix is: ")
print(Q)

Q_row_sum = np.sum(Q, axis = 1)
Q_percentage = np.multiply(np.divide(Q, np.transpose(Q_row_sum)), 100)
print("Q Percentage is: ")
print(np.round(Q_percentage, 2))

np.savetxt(test_root_folder + 'Q_Matrix_249_gestures.out', Q, fmt = '%d', newline='\n', delimiter=',')
np.savetxt(test_root_folder + 'Q_Percentage_249_gesutres.out', Q_percentage, fmt = '%.2f', delimiter=',')

# True Predictions: 5068, False Predictions: 932 in average time: 57.1725 ms.   
# Results Quantified!!!
# 6000/6000 [==============================] - 26s 4ms/step
# Model Accuracy:  [0.6224436934391657, 0.8675]
