import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import sys
from glob import glob
import cv2
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

return_flag = None

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def create_all_hands_dataset(root_folder, num_of_dataset, samples, time_steps, \
rescale_h_w, left_files, right_files):
    all_hand = [None] * 2
    for hand in range(0, 2):
    # for hand in range(1, 2):
        names = [None] * num_of_dataset
        if hand == 0:
            files = left_files
            name = 'left'
            filename = root_folder+'full/left.dat'
        else:
            files = right_files
            name = 'right'
            filename = root_folder+'full/right.dat'
        for i in range(0, np.size(files)):
            names[i]  = np.memmap(files[i], dtype='float32', mode='r', shape=(int(samples[i]), time_steps, rescale_h_w, rescale_h_w, 3))

        print("Creating a full memmap array for " + name + " hand of all datasets.")
        all_hand[hand] = np.memmap(filename, dtype='float32', mode='r+', shape=(int(np.sum(samples)), time_steps, rescale_h_w, rescale_h_w, 3))

        for i in range(0, np.size(files)-1):
            if i == 0:
                start_index = int(samples[i])
                end_index = int(samples[i]) + int(samples[i+1])
            else:
                start_index = end_index
                end_index = end_index + int(samples[i+1])
            print("Filling " + name + " hand array " + str(i+2) + " out of " + str(int(num_of_dataset)) + ".    ", end ="\r", flush=True)
            all_hand[hand][start_index:end_index,:,:,:,:] = names[i+1]
        print("")


def get_hand_dataset_filenames(root_folder):
    data_files = glob(root_folder+'/*.dat')
    data_files.sort()
    short_files = data_files.copy()

    left_find  = []
    right_find = []

    for i in range(0, np.size(short_files)):
        short_files[i] = data_files[i].replace(root_folder, '')
        left_find_temp = short_files[i].find('left')
        if left_find_temp != -1:
            left_find.append(i)
        right_find_temp = short_files[i].find('right')
        if right_find_temp != -1:
            right_find.append(i)

    left_files = [data_files[i] for i in left_find]
    right_files = [data_files[i] for i in right_find]
    left_files.sort()
    right_files.sort()
    return left_files, right_files

def normalize_hand_vector(save_drive, root_folder, full_filename, hand_vector, name, rescale_h_w, duplicate):
    short_name = full_filename.replace(root_folder, '')
    norm_filename = short_name.replace('.h5', '_' + name + '.dat')
    norm_filename = save_drive+norm_filename
    print("Now normalizing " + name + " hand vector")

    hand_vector = np.moveaxis(hand_vector, -1, 0)
    hand_vector = np.moveaxis(hand_vector, -1, 1)

    norm_vector = np.memmap(norm_filename, dtype='float32', mode='w+', shape=(np.shape(hand_vector)[0], np.shape(hand_vector)[1], rescale_h_w, rescale_h_w, 3))
    resized_image = np.zeros((rescale_h_w,rescale_h_w,3))

    for i in range(0, np.shape(hand_vector)[0]):
        for j in range(0, np.shape(hand_vector)[1]):
            resized_image = cv2.resize(hand_vector[i,j,:,:,:],(int(rescale_h_w),int(rescale_h_w)))
            norm_vector[i,j,:,:,:] = resized_image / 255.
            norm_vector[i,j,:,:,:] = norm_vector[i,j,:,:,:].astype('float32')
        print(str("Normalizing Sample " + str(i+1) + " out of " + str(np.shape(hand_vector)[0]) + " samples."), end ="\r", flush=True)
    print("")
    if duplicate:
        print("Duplicating the first " + name + " hand vector for later use.")
        norm_vector_dup = np.memmap(save_drive+'full/'+name+'.dat', dtype='float32', mode='w+', shape=(np.shape(hand_vector)[0], np.shape(hand_vector)[1], rescale_h_w, rescale_h_w, 3))
        print("Filling " + name + " duplicate vector.")
        norm_vector_dup[:] = norm_vector[:]

def normalize_X(X_vector, label):
    samples, time_steps, feature_set = np.shape(X_vector)
    print("Total samples in all_x: ", samples)
    for_minmax = X_vector.reshape(samples*time_steps, feature_set)
    zero_array = np.zeros(np.shape(for_minmax))
    zero_indices = np.where(~for_minmax.any(axis=1))[0]
    non_zero_indices = np.where(for_minmax.any(axis=1))[0]
    for_minmax_wo_zero = np.delete(for_minmax, zero_indices, axis=0)

    if label == 'train':
        Xscaler = MinMaxScaler()
        Xscaler.fit(for_minmax_wo_zero)
        for_minmax = Xscaler.transform(for_minmax_wo_zero)
        train_scaler_filename = "train_scalar.save"
        joblib.dump(Xscaler, train_scaler_filename)

    if label == 'valid' or label == 'test':
        train_scaler_filename = "train_scalar.save"
        Xscaler = joblib.load(train_scaler_filename)
        for_minmax = Xscaler.transform(for_minmax_wo_zero)

    zero_array[non_zero_indices] = for_minmax
    # # # new = Xscaler.inverse_transform(for_minmax)
    # # # X_inversed = new.reshape(samples, time_steps, feature_set)
    X_vector_normed = zero_array.reshape(samples, time_steps, feature_set)
    return X_vector_normed

def read_dataset(root_folder, full_filename):
    read_h5 = h5py.File(full_filename, 'r')
    short_name = full_filename.replace(root_folder, '')
    # read_h5 = h5py.File("./gesture_5_dataset_small.h5", 'r')
    print("Now reading X, y vectors and other information from " + short_name)
    X_vector = read_h5['X_vector'][:]
    # y_vector = read_h5['y_vector'][:]
    one_y_vector = read_h5['one_y_vector'][:]
    gesture_num = read_h5['gesture_num'][()]
    print("Now reading X_left_hand_vector from " + short_name)
    X_left_hand_vector = read_h5['X_left_hand_vector'][:]
    print("Now reading X_right_hand_vector from " + short_name)
    X_right_hand_vector = read_h5['X_right_hand_vector'][:]
    read_h5.close()
    # X_left_hand_vector = 0
    # X_right_hand_vector = 0
    return X_vector, X_left_hand_vector, X_right_hand_vector, \
    one_y_vector, gesture_num

def get_gesture_filenames(set_name, vid_filenames, gesture_descriptors_location):
    gesture_filenames = []
    for i in range(0, np.size(vid_filenames)):
        vid_filenames[i][0] = vid_filenames[i][0].replace(set_name, '')
        vid_filenames[i][0] = vid_filenames[i][0].replace('.avi', '.h5')
        temp_gesture = gesture_descriptors_location + vid_filenames[i][0]
        gesture_filenames = np.append(gesture_filenames, temp_gesture)
    return gesture_filenames


def create_fixed_size_vectors(count, max_videos, gesture_count, \
gesture_filenames, orig_vid_max_length, rescale_h_w, extra_padding):

    length_of_samples = []
    gesture_delete_indices = []
    y_vector = []
    one_y_vector = []
    X_vector = []
    resized_image = np.zeros((rescale_h_w,rescale_h_w,3))
    empty_image = np.zeros((int(rescale_h_w), int(rescale_h_w), 3), dtype = 'uint8')
    num_of_current_vids =  np.size(gesture_filenames)
    for i in range(0, num_of_current_vids):
    # for i in range(0, 100):
        # print("")
        # print("Reading data from: ", gesture_filenames[i])
        read_h5 = h5py.File(gesture_filenames[i], 'r')
        read_cropped_left_vector  = read_h5['cropped_left_vector'][:]
        read_cropped_left_vector  = np.moveaxis(read_cropped_left_vector, 0, -1)
        read_cropped_right_vector = read_h5['cropped_right_vector'][:]
        read_cropped_right_vector = np.moveaxis(read_cropped_right_vector, 0, -1)
        read_neck_feature_vector  = read_h5['neck_feature_vector'][:]
        read_h5.close()
        # print("Reading data OK!")
        # np.allclose(total_images, read_train_images)
        # print('Size of cropped_left_vector: ', np.shape(read_cropped_left_vector))
        # print('Size of cropped_right_vector: ', np.shape(read_cropped_right_vector))
        # print('Size of neck_feature_vector: ', np.shape(read_neck_feature_vector))

        if np.size(read_cropped_left_vector) != 0:
            length_of_current_video = np.shape(read_neck_feature_vector)[1]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # need to remove equality condition of shapes of left and right hand vector after updating (rerun create descriptor) the dataset.
            if length_of_current_video < (orig_vid_max_length + 1) and np.shape(read_cropped_left_vector)[3] == np.shape(read_cropped_right_vector)[3]:
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # print("Old length of video", np.shape(read_neck_feature_vector)[1])
                length_of_samples = np.append(length_of_samples, length_of_current_video)
                padding_times = (orig_vid_max_length + extra_padding) - length_of_current_video
                # print("New length of video", np.shape(read_neck_feature_vector)[1])

                # >>> Substitute of Matlab repmat
                N = empty_image[:,:,:,np.newaxis]
                # <<< Substitute of Matlab repmat
                if padding_times % 2 == 0:
                    half_padding_times = int(padding_times / 2)

                    zero_padding_neck = np.zeros((np.shape(read_neck_feature_vector)[0], half_padding_times))
                    read_neck_feature_vector = np.append(zero_padding_neck, read_neck_feature_vector, axis=1)
                    read_neck_feature_vector = np.append(read_neck_feature_vector, zero_padding_neck, axis=1)

                    zero_padding_hand = np.tile(N,(1,1,1,half_padding_times))
                    cropped_left_vector  = np.append(zero_padding_hand, read_cropped_left_vector,  axis=3)
                    cropped_left_vector  = np.append(cropped_left_vector, zero_padding_hand,  axis=3)
                    cropped_right_vector = np.append(zero_padding_hand, read_cropped_right_vector, axis=3)
                    cropped_right_vector = np.append(cropped_right_vector, zero_padding_hand, axis=3)
                else:
                    half_padding_times = int((padding_times + 1) / 2)
                    if half_padding_times == 1:
                        zero_padding_neck_post = np.zeros((np.shape(read_neck_feature_vector)[0], half_padding_times))
                        read_neck_feature_vector = np.append(read_neck_feature_vector, zero_padding_neck_post, axis=1)

                        zero_padding_post = np.tile(N,(1,1,1,half_padding_times))
                        cropped_left_vector  = np.append(read_cropped_left_vector, zero_padding_post,  axis=3)
                        cropped_right_vector = np.append(read_cropped_right_vector, zero_padding_post, axis=3)
                    else:
                        zero_padding_neck_pre = np.zeros((np.shape(read_neck_feature_vector)[0], half_padding_times-1))
                        zero_padding_neck_post = np.zeros((np.shape(read_neck_feature_vector)[0], half_padding_times))
                        read_neck_feature_vector = np.append(zero_padding_neck_pre, read_neck_feature_vector, axis=1)
                        read_neck_feature_vector = np.append(read_neck_feature_vector, zero_padding_neck_post, axis=1)

                        zero_padding_pre = np.tile(N,(1,1,1,half_padding_times-1))
                        zero_padding_post = np.tile(N,(1,1,1,half_padding_times))
                        cropped_left_vector  = np.append(zero_padding_pre, read_cropped_left_vector,   axis=3)
                        cropped_left_vector  = np.append(cropped_left_vector, zero_padding_post,  axis=3)
                        cropped_right_vector = np.append(zero_padding_pre, read_cropped_right_vector,  axis=3)
                        cropped_right_vector = np.append(cropped_right_vector, zero_padding_post, axis=3)
            else:
                trunc_count = length_of_current_video - orig_vid_max_length
                if trunc_count % 2 == 0:
                    trunc_count_half = int(trunc_count / 2)
                    cropped_left_vector  = read_cropped_left_vector[:,:,:,trunc_count_half:-trunc_count_half]
                    cropped_right_vector = read_cropped_right_vector[:,:,:,trunc_count_half:-trunc_count_half]
                    read_neck_feature_vector = read_neck_feature_vector[:,trunc_count_half:-trunc_count_half]
                else:
                    trunc_count_half = int((trunc_count + 1) / 2)
                    if trunc_count_half == 1:
                        cropped_left_vector  = read_cropped_left_vector[:,:,:,trunc_count_half:]
                        cropped_right_vector = read_cropped_right_vector[:,:,:,trunc_count_half:]
                        read_neck_feature_vector = read_neck_feature_vector[:,trunc_count_half:]
                    else:
                        cropped_left_vector  = read_cropped_left_vector[:,:,:,trunc_count_half:-(trunc_count_half-1)]
                        cropped_right_vector = read_cropped_right_vector[:,:,:,trunc_count_half:-(trunc_count_half-1)]
                        read_neck_feature_vector = read_neck_feature_vector[:,trunc_count_half:-(trunc_count_half-1)]

            cropped_left_vector  = cropped_left_vector[:,:,:,:,np.newaxis]
            cropped_right_vector = cropped_right_vector[:,:,:,:,np.newaxis]

            if len(X_vector) == 0:
                X_vector = read_neck_feature_vector
                X_left_hand_vector  = cropped_left_vector
                X_right_hand_vector = cropped_right_vector
            else:
                X_vector = np.dstack((X_vector, read_neck_feature_vector))
                X_left_hand_vector  = np.concatenate((X_left_hand_vector,  cropped_left_vector),  axis=4)
                X_right_hand_vector = np.concatenate((X_right_hand_vector, cropped_right_vector), axis=4)

            # y_current_zeros = np.zeros((1, length_of_current_video))
            # y_current_ones = np.ones((1, orig_vid_max_length - length_of_current_video))
            # y_current = np.append(y_current_zeros, y_current_ones)
            # y_current_reshaped = np.reshape(y_current, (1, np.size(y_current)))
            #
            # if len(y_vector) == 0:
            #     y_vector = y_current_reshaped
            # else:
            #     y_vector = np.append(y_vector, y_current_reshaped, axis=0)
            # print("length_of_current_video: ", length_of_current_video)
            # print("length_of_current_video: ", length_of_current_video)
            one_y_vector = np.append(one_y_vector, gesture_count)

            # print("Shape of one_y_vector: ", np.shape(one_y_vector))
            # print("Size of X_left vector:  ", np.shape(X_left_hand_vector))
            # print("Size of X_right vector: ", np.shape(X_right_hand_vector))
            print(str(i+1) + "/" + str(num_of_current_vids) + " from current gesture and " + \
            str(count) + "/" + str(int(max_videos)) + " videos processed in all.", end ="\r", flush=True)
            count = count + 1
            # else:
            #     count = count + 1
            #     gesture_delete_indices = np.append(gesture_delete_indices, i)
            # print("")
    print("")
    return count, X_vector, X_left_hand_vector, X_right_hand_vector, \
    y_vector, one_y_vector, gesture_delete_indices

def draw_skeleton(display_image, mid_array_keypoints, line_indices_1, line_indices_2):
    for m in range(0, np.size(mid_array_keypoints, 0) - 15): # Only top 14 keypoints are drawn
        if m == 8 or m == 9:
            continue
        # cv2.circle(output_image,(int(mid_array_keypoints[i,0]), int(mid_array_keypoints[i,1])), 3, (0,0,255), -1)
        if np.count_nonzero(mid_array_keypoints[m]):
            cv2.circle(display_image,(int(mid_array_keypoints[m,0]), int(mid_array_keypoints[m,1])), 3, [0,0,255], -1)

    for m in range(0, np.size(line_indices_1, 0)):
        # temp_1 = np.array((int(mid_array_keypoints[line_indices_1[m],0]), int(mid_array_keypoints[line_indices_1[m],1])))
        # temp_2 = np.array((int(mid_array_keypoints[line_indices_2[m],0]), int(mid_array_keypoints[line_indices_2[m],1])))
        temp_1 = np.round(mid_array_keypoints[line_indices_1[m],:2])
        temp_2 = np.round(mid_array_keypoints[line_indices_2[m],:2])
        if (np.count_nonzero(temp_1) and np.count_nonzero(temp_2)): # To avoid drawing lines if one of the point is (0, 0)
            # cv2.line(output_image, (temp_1[0], temp_1[1]), (temp_2[0], temp_2[1]), (0,0,255), 2, -1)
            cv2.line(display_image, (int(temp_1[0]), int(temp_1[1])), (int(temp_2[0]), int(temp_2[1])), (0,0,255), 2, -1)
    # cv2.circle(display_image,(int(lh_keypts[0,0]), int(lh_keypts[0,1])), 10, (0,0,255), -1)


def show_skeleton_array(read_neck_feature_vector):
    out_img_rows = 1080
    out_img_cols = 1440
    output_img = np.zeros((int(out_img_rows), int(out_img_cols), 3), dtype = 'uint8')
    line_indices_1 = [1, 1, 1, 2, 3, 5, 6]
    line_indices_2 = [0, 2, 5, 3, 4, 6, 7]
    upper_skeleton = read_neck_feature_vector[:16,:]
    print("before: ", upper_skeleton[:,0])
    upper_skeleton = upper_skeleton.reshape(8, 2, np.shape(read_neck_feature_vector)[1])
    print("after: ", upper_skeleton[:,:,0])
    print(np.shape(upper_skeleton))
    draw_skeleton(output_img, upper_skeleton[:,:,0]+500, line_indices_1, line_indices_2)
    cv2.imshow('skeleton test', output_img)
    k = cv2.waitKey(0)
    if k==27:    # Esc key to stop
        cv2.destroyAllWindows()
        print("Escape pressed... Program Closing!")
        exit()
    # for m in range(0, np.size(mid_array_keypoints, 0) - 15): # Only top 14 keypoints are drawn
    #     if m == 8 or m == 9:
    #         continue
    #     # cv2.circle(output_image,(int(mid_array_keypoints[i,0]), int(mid_array_keypoints[i,1])), 3, (0,0,255), -1)
    #     if np.count_nonzero(mid_array_keypoints[m]):
    #         cv2.circle(display_image,(int(mid_array_keypoints[m,0]), int(mid_array_keypoints[m,1])), 3, [0,0,255], -1)
    #
    # for m in range(0, np.size(line_indices_1, 0)):
    #     # temp_1 = np.array((int(mid_array_keypoints[line_indices_1[m],0]), int(mid_array_keypoints[line_indices_1[m],1])))
    #     # temp_2 = np.array((int(mid_array_keypoints[line_indices_2[m],0]), int(mid_array_keypoints[line_indices_2[m],1])))
    #     temp_1 = np.round(mid_array_keypoints[line_indices_1[m],:2])
    #     temp_2 = np.round(mid_array_keypoints[line_indices_2[m],:2])
    #     if (np.count_nonzero(temp_1) and np.count_nonzero(temp_2)): # To avoid drawing lines if one of the point is (0, 0)
    #         # cv2.line(output_image, (temp_1[0], temp_1[1]), (temp_2[0], temp_2[1]), (0,0,255), 2, -1)
    #         cv2.line(output_img, (int(temp_1[0]), int(temp_1[1])), (int(temp_2[0]), int(temp_2[1])), (0,0,255), 2, -1)
    # # cv2.circle(display_image,(int(lh_keypts[0,0]), int(lh_keypts[0,1])), 10, (0,0,255), -1)
    #

def show_extra_skeleton(mid_array_keypoints):
    out_img_rows = 1080
    out_img_cols = 1440
    output_img = np.zeros((int(out_img_rows), int(out_img_cols), 3), dtype = 'uint8')
    line_indices_1 = [1, 1, 1, 2, 3, 5, 6]
    line_indices_2 = [0, 2, 5, 3, 4, 6, 7]
    for m in range(0, np.size(mid_array_keypoints, 0) - 15): # Only top 14 keypoints are drawn
        if m == 8 or m == 9:
            continue
        # cv2.circle(output_image,(int(mid_array_keypoints[i,0]), int(mid_array_keypoints[i,1])), 3, (0,0,255), -1)
        if np.count_nonzero(mid_array_keypoints[m]):
            cv2.circle(output_img,(int(mid_array_keypoints[m,0]), int(mid_array_keypoints[m,1])), 3, [0,0,255], -1)

    for m in range(0, np.size(line_indices_1, 0)):
        # temp_1 = np.array((int(mid_array_keypoints[line_indices_1[m],0]), int(mid_array_keypoints[line_indices_1[m],1])))
        # temp_2 = np.array((int(mid_array_keypoints[line_indices_2[m],0]), int(mid_array_keypoints[line_indices_2[m],1])))
        temp_1 = np.round(mid_array_keypoints[line_indices_1[m],:2])
        temp_2 = np.round(mid_array_keypoints[line_indices_2[m],:2])
        if (np.count_nonzero(temp_1) and np.count_nonzero(temp_2)): # To avoid drawing lines if one of the point is (0, 0)
            # cv2.line(output_image, (temp_1[0], temp_1[1]), (temp_2[0], temp_2[1]), (0,0,255), 2, -1)
            cv2.line(output_img, (int(temp_1[0]), int(temp_1[1])), (int(temp_2[0]), int(temp_2[1])), (0,0,255), 15, -1)
    # cv2.circle(display_image,(int(lh_keypts[0,0]), int(lh_keypts[0,1])), 10, (0,0,255), -1)
    # cv2.imshow("test", output_img)
    return output_img

def press(event):
    # print('press', event.key)
    global return_flag
    sys.stdout.flush()
    if event.key == 'x':
        print("Exiting Program!")
        sys.exit()
    if event.key == 'w':
        return_flag = event.key

def fill_image(X, num_of_images_in_x, num_of_images_in_y, \
height_of_single_image, width_of_single_image, frame_output_height, \
frame_output_width, label):

    location_x_text = int(frame_output_width / 100 * 90)
    location_y_text = int(frame_output_height / 100 * 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 3

    total_num_images = np.shape(X)[0]
    count = 0
    empty_image = np.ones((int(height_of_single_image), int(width_of_single_image), 3), dtype = 'uint8')
    empty_image[:,:,0] = empty_image[:,:,0] * 255
    empty_image[:,:,2] = empty_image[:,:,2] * 255

    output_image  = np.ones((int(frame_output_height), int(frame_output_width), 3), dtype = 'uint8')

    for y in range(0, num_of_images_in_y):
        for x in range(0, num_of_images_in_x):
            # print(np.array2string(Y[:,count]))
            if count < total_num_images:
                image = X[count,:,:,:]
            else:
                image = empty_image
            image = image[:,:,::-1]
            image = cv2.normalize(image, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            resized_image = cv2.resize(image, (width_of_single_image, height_of_single_image))
            x_position = x * int(width_of_single_image) - 1
            if x_position < 0:
                x_position = 0
            y_position = y * int(height_of_single_image) - 1
            if y_position < 0:
                y_position = 0
            # cv2.imshow("Output Photo", output_image)
            output_image[y_position:y_position + height_of_single_image, x_position:x_position + width_of_single_image] = resized_image[:,:]
            count += 1
            cv2.waitKey(1)
    if not (label is None):
        cv2.putText(output_image,str(int(label)) ,(location_x_text,location_y_text), font, font_size,(0,0,255),font_size*2,cv2.LINE_AA)
    return output_image

def dataset_show_image(X_left, X_right, num_of_images_in_x, *args, **kwargs):
    label = kwargs.get('label', None)
    Show = kwargs.get('Show', None)

    init_output_width = 1920

    width_of_single_image = int(init_output_width / num_of_images_in_x)
    height_of_single_image = width_of_single_image
    remainder = np.remainder(np.shape(X_left)[0] , num_of_images_in_x)
    if remainder:
        num_of_images_in_y = int(np.shape(X_left)[0] / num_of_images_in_x + 1)
    else:
        num_of_images_in_y = int(np.shape(X_left)[0] / num_of_images_in_x)

    # num_of_images_in_y = 5
    total_number_of_embedding_images = num_of_images_in_x * num_of_images_in_y

    frame_output_width = width_of_single_image * num_of_images_in_x
    frame_output_height = height_of_single_image * num_of_images_in_y

    left_image = fill_image(X_left, num_of_images_in_x, num_of_images_in_y, height_of_single_image, \
                width_of_single_image, frame_output_height, frame_output_width, label)

    right_image = fill_image(X_right, num_of_images_in_x, num_of_images_in_y, height_of_single_image, \
                width_of_single_image, frame_output_height, frame_output_width, label)
    if Show:
        # my_dpi = 90
        # # fig =  plt.figure(figsize=(int(frame_output_width)/my_dpi, int(frame_output_height)/my_dpi), dpi=my_dpi)
        # fig =  plt.figure(figsize=(8,4), dpi=my_dpi)
        fig, axs = plt.subplots(2, figsize=(8,10))
        fig.canvas.mpl_connect('key_press_event', press)
        plt.suptitle("q: close window, x: close the program, w then q: skip show")
        axs[0].title.set_text("Left Hand Images")
        axs[1].title.set_text("Right Hand Images")
        axs[0].axis('off')
        axs[1].axis('off')
        axs[0].imshow(left_image[:,:,::-1])
        axs[1].imshow(right_image[:,:,::-1])
        plt.rcParams["keymap.quit"] = ['ctrl+w', 'cmd+w', 'q']
        plt.show()
    return return_flag, left_image, right_image

def read_skeleton_data(root_folder, name_of_dataset, *args, **kwargs):
    type = kwargs.get('type', None)
    if not type:
        type = ''
    read_h5 = h5py.File(root_folder+name_of_dataset, 'r')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print('Reading all_X_normed ' + type + '...')
    # all_X_normed = read_h5['all_X_normed'][:]
    print('Reading y_one_hot ' +type+'...')
    y_one_hot = read_h5['y_one_hot'][:]
    print('Reading real_ges_vals ' +type+'...')
    real_ges_vals = read_h5['real_ges_vals'][:]
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # print('Size of all_X_normed '+type + ' is:  ', np.shape(all_X_normed))
    print('Size of y_one_hot '+type+' is:     ', np.shape(y_one_hot))
    print('Size of real_ges_vals '+type+' is: ', np.shape(real_ges_vals))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    read_h5.close()
    return y_one_hot, real_ges_vals

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

def read_standard_normed_pose(root_folder, name_of_dataset, *args, **kwargs):
    type = kwargs.get('type', None)
    if not type:
        type = ''
    read_h5 = h5py.File(root_folder+name_of_dataset, 'r')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Reading all_X_standard_normed ' + type + '...')
    all_X_standard_normed = read_h5['all_X_standard_normed'][:]
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Size of all_X_standard_normed '+type + ' is:  ', np.shape(all_X_standard_normed))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    read_h5.close()
    return all_X_standard_normed
