# from glob import glob
import numpy as np
import time
from glob import glob
import os
# from keras.models import load_model
# from keras import optimizers
from rnn_utils import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tempfile import mkdtemp
import os.path as path

set_name = 'test'

root_folder = "/media/osama/My-book/IsoGD/IsoGD_phase_1/"+set_name+"_dataset_files/" # with slash in the end
dataset_files = glob(root_folder+'*.h5')
dataset_files.sort()
save_drive = '/media/osama/My-book/IsoGD/IsoGD_phase_1/'+set_name+'_ready/'
os.makedirs(save_drive+'full', exist_ok=True )
num_of_dataset = np.size(dataset_files)
samples = np.zeros(num_of_dataset)
one_neuron_y = []
real_ges_vals = []

np.random.seed(7)

# Process data

rescale_h_w = 224
duplicate = 1

# for i in range(num_of_dataset - 16, num_of_dataset):
#     print(dataset_files[i])

### !!! gesture-50 is problematic
print("Total Number of Gestures Detected: ", num_of_dataset)
# for i in range(0, 10):
for i in range(0, num_of_dataset):
    if i != 0:
        duplicate = 0
    X_vector, X_left_hand_vector, X_right_hand_vector, \
    one_y_vector, gesture_num = read_dataset(root_folder, dataset_files[i])

    orig_X = X_vector.copy()
    X_vector = np.moveaxis(X_vector, 1, -1)
    X_vector = np.moveaxis(X_vector, 0, -1)

    samples[i], time_steps, feature_set = np.shape(X_vector)

    # X_vector_normed, samples[i], time_steps, feature_set = normalize_X(X_vector, label='valid')
    # print("Number of Samples in gesture " + str(gesture_num) + " is " + str(samples[i]))
    # if i == 0:
    #     all_X_normed = X_vector_normed
    # else:
    #     all_X_normed = np.vstack((all_X_normed, X_vector_normed))

    print("Number of Samples in gesture " + str(gesture_num) + " is " + str(samples[i]))
    if i == 0:
        all_X = X_vector
    else:
        all_X = np.vstack((all_X, X_vector))

    print("Shape of one_y_vector: ", np.shape(one_y_vector))
    new_one_y = np.ones(np.size(one_y_vector)) * i
    one_neuron_y = np.append(one_neuron_y, new_one_y)
    one_neuron_y = one_neuron_y.astype('int')

    real_ges_vals = np.append(real_ges_vals, gesture_num)
    normalize_hand_vector(save_drive, root_folder, dataset_files[i], X_left_hand_vector, \
    'left', rescale_h_w, duplicate)
    normalize_hand_vector(save_drive, root_folder, dataset_files[i], X_right_hand_vector, \
    'right', rescale_h_w, duplicate)
    left_files, right_files = get_hand_dataset_filenames(save_drive)

all_X_normed = normalize_X(all_X, label=set_name)
print("Shape of all_X_normed: ", np.shape(all_X_normed))
y_one_hot = convert_to_one_hot(one_neuron_y, num_of_dataset).T

create_all_hands_dataset(save_drive, num_of_dataset, samples, time_steps, \
rescale_h_w, left_files, right_files)

# # >>> Splitting turned off
# print("Data processing finished!")
# print("Splitting Data in to train and test sets...")
#
# total_samples = int(np.sum(samples))
# print("Total number of samples: ", total_samples)
#
# split_index_list = np.arange(total_samples)
# np.random.shuffle(split_index_list)
# split_ratio = 0.2
# test_size = int(total_samples*0.2)
# train_size = total_samples - test_size
#
# print("train size: ", train_size)
# print("test size : ", test_size)
#
# train_indices = split_index_list[:train_size]
# test_indices = split_index_list[train_size:train_size+test_size]
#
# print("Fill in other matrices")
#
# xTrain = all_X_normed[train_indices, :, :]
# xTest = all_X_normed[test_indices, :, :]
# y_one_hot_train = y_one_hot[train_indices, :]
# y_one_hot_test = y_one_hot[test_indices, :]
#
# print("Shape of xTrain: ", np.shape(xTrain))
# print("Shape of xTest: ", np.shape(xTest))
# print("Shape of y_one_hot_train: ", np.shape(y_one_hot_train))
# print("Shape of y_one_hot_test: ", np.shape(y_one_hot_test))
# print("Writing the created vectors...")
# file_h5 = h5py.File(save_drive+'full/top'+str(num_of_dataset)+'_gestures_normed.h5', 'w')
# file_h5.create_dataset('xTrain', data = xTrain)
# file_h5.create_dataset('xTest', data = xTest)
# file_h5.create_dataset('y_one_hot_train', data = y_one_hot_train)
# file_h5.create_dataset('y_one_hot_test', data = y_one_hot_test)
# file_h5.create_dataset('real_ges_vals', data = real_ges_vals)
# file_h5.close()
# print("Writing done...")
# # <<< Splitting turned off

print("Writing the created vectors...")
file_h5 = h5py.File(save_drive+'full/top'+str(num_of_dataset)+'_zp_gestures_together_normed.h5', 'w')
file_h5.create_dataset('all_X_normed', data = all_X_normed)
file_h5.create_dataset('y_one_hot', data = y_one_hot)
file_h5.create_dataset('real_ges_vals', data = real_ges_vals)
file_h5.close()
print("Writing done...")

print("Writing the created vectors...")
file_h5 = h5py.File(save_drive+'full/top'+str(num_of_dataset)+'_zp_gestures_original.h5', 'w')
file_h5.create_dataset('all_X', data = all_X)
file_h5.close()
print("Writing done...")

# print("")
#
# y_vector = y_vector.reshape(np.shape(y_vector)[0], 40, 1)
#
# y_one = []
#
# for i in range(0, np.shape(y_vector)[0]):
#     # print(y_vector[i,:].reshape(1, np.size(y_vector[i,:])))
#     if np.count_nonzero(y_vector[i,:]):
#         non_zero_count = non_zero_count + 1
#         y_one = np.append(y_one, 1)
#     else:
#         zero_count = zero_count + 1
#         y_one = np.append(y_one, 0)
#
# y_one = y_one.reshape(np.size(y_one), 1)
# print("Size of y_one: ", np.shape(y_one))
#
# print("Ones  in y_vector: ", non_zero_count)
# print("Zeros in y_vector: ", zero_count)
#
#
# # xTrain, xTest, y_one_Train, y_one_Test = train_test_split(X_vector_normed, y_one, test_size = 0.2, random_state = 7, shuffle = True)
#

#
# # for i in range(0, np.shape(X_lhvn_Train)[0]):
# #     stop = dataset_show_image(X_lhvn_Train[i,:,:,:], X_rhvn_Train[i,:,:,:], 10, Show=True)
# #     if stop == 'w':
# #         print("I broke the loop")
# #         break
# #
# # for i in range(0, np.shape(X_rhvn_Test)[0]):
# #     stop = dataset_show_image(X_rhvn_Test[i,:,:,:], X_rhvn_Test[i,:,:,:], 10, Show=True)
# #     if stop == 'w':
# #         print("I broke the loop")
# #         break
#
# # print("xTrain is: ", xTrain)
# # print("xTest is: ", xTest)
# # print("y_one_Train is: ", y_one_Train)
# # print("y_one_Test is: ", y_one_Test)
#
