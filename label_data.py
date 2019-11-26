import csv
import numpy as np
from glob import glob
import h5py
import cv2
import os

from rnn_utils import *
np.set_printoptions(suppress=True)

set_name = 'train'

with open('/media/osama/My-book/IsoGD/IsoGD_phase_1/'+set_name+'.txt') as inf:
    reader = csv.reader(inf, delimiter=" ")
    data = list(zip(*reader))
    color = data[0]
    depth = data[1]
    label = data[2]

color = np.asarray(color).transpose()
color = np.reshape(color, (np.size(color), 1))

# depth = np.asarray(depth).transpose()
# depth = np.reshape(depth, (np.size(depth), 1))
#
label = list(map(int, label))
label = np.asarray(label)
# label = np.reshape(label, (np.size(label), 1))

print("Shape of color: ", np.shape(color))
# print("Shape of depth: ", np.shape(depth))
print("Shape of label: ", np.shape(label))

# print(color)
# print(depth)
# print(label)
# print("Max number of gestures: ", max(label))

# for i in range(1, max(label)):
#     result = np.where(label == i)
#     # print(str("Gesture [" + str(i) + "] appears " + str(np.size(result)) + " times."))
#     if i == 5:

num_of_videos_each_label = []
for i in range(0, max(label)):
    label_index = np.where(label == i+1)
    num_of_videos_each_label = np.append(num_of_videos_each_label, np.size(label_index))

max_videos_of_all = int(np.amax(num_of_videos_each_label))
print("Max videos: ", max_videos_of_all)

print("Mean number of videos: ", np.mean(num_of_videos_each_label))
print("Total number of gestures: ", np.shape(num_of_videos_each_label)[0])
sorted_indices = np.unravel_index(np.argsort(-num_of_videos_each_label, axis=None), num_of_videos_each_label.shape)

print(num_of_videos_each_label[sorted_indices])
print(sorted_indices)

root_folder = '/media/osama/My-book/IsoGD/IsoGD_phase_1/'
gesture_descriptors_location = root_folder + set_name + '_Gesture_descriptors'
dataset_save_root = root_folder + set_name + '_dataset_files'
os.makedirs(dataset_save_root, exist_ok=True )
Top_gestures_considered = 249
label_indices_top = []
num_of_videos_top = []

for i in range(0, Top_gestures_considered):
    temp = np.where(label == sorted_indices[0][i]+1)
    label_indices_top = np.append(label_indices_top, np.where(label == sorted_indices[0][i]+1))
    num_of_videos_top = np.append(num_of_videos_top, num_of_videos_each_label[sorted_indices[0][i]])
    # print(str("Number of videos of gesture " + str(sorted_indices[0][i]+1) + " are: " + str(np.size(label_indices_top[i][:]))))

print(num_of_videos_top)
print("Total num of videos: ", int(np.sum(num_of_videos_top)))

start_index = 0
orig_vid_max_length = 40
extra_padding = 0 # adding extra zeros (everywhere) after gesture is finished
rescale_h_w = 224
count = 1

for i in range(0, Top_gestures_considered):
    gesture_num = sorted_indices[0][i] + 1
    if i == 0:
        start_index = 0
    else:
        start_index = start_index + int(num_of_videos_top[i-1])
    end_index =  start_index + int(num_of_videos_top[i])
    current_vid_indices = label_indices_top[start_index:end_index].astype(int)
    current_vid_filenames = color[current_vid_indices]
    print("Processing Videos for gesture " + str(gesture_num))
    current_ges_filenames = get_gesture_filenames(set_name, current_vid_filenames, gesture_descriptors_location)
    count, X_vector, X_left_hand_vector, X_right_hand_vector, \
    _, one_y_vector, _ = create_fixed_size_vectors(count, np.sum(num_of_videos_top), i, \
    current_ges_filenames, orig_vid_max_length, rescale_h_w, extra_padding)
    dataset_filename = dataset_save_root + "/gesture_" + str(gesture_num) + "_dataset.h5"
    print("Saving data in " + dataset_filename)
    file_h5 = h5py.File(dataset_filename, 'w')
    file_h5.create_dataset('X_vector', data = X_vector)
    file_h5.create_dataset('X_left_hand_vector', data = X_left_hand_vector)
    file_h5.create_dataset('X_right_hand_vector', data = X_right_hand_vector)
    file_h5.create_dataset('one_y_vector', data = one_y_vector)
    file_h5.create_dataset('gesture_num', data = gesture_num)
    file_h5.close()

# gesture_5 = color[result_5]
#
# # >>> getting false samples from dataset other than considered gesture (which is 5 for now)
#
# num_of_false_samples = 1500
#
# other_labels = np.delete(label, result_5)
# other_color = np.delete(color, result_5)
#
# print("Size of remaining gestures: ", np.shape(other_labels))
#
# np.random.seed(1)
# false_set_indices = np.random.choice(range(np.size(other_labels)), num_of_false_samples, replace=False) # False for unique samples
# false_labels = other_labels[false_set_indices]
# false_color = other_color[false_set_indices]
#
# result_4 = np.where(false_labels == 4)
# print(false_color[result_4])
# print("result _4: ", result_4)
