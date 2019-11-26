import json
import objectpath
import numpy as np
import cv2 # delete libqt5x11extras5 if you are facing Qt errors.
from glob import glob
from math import pi, sqrt, exp, cos, sin, atan2
from itertools import combinations
from numpy import linalg as la
from mmap import ACCESS_READ, mmap, ALLOCATIONGRANULARITY
import array
import time
import os
import re
import h5py

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
# from keras.models import load_model
# from keras import optimizers

def Gaussian_filter(filter_size = 5, sigma = 1):
    range_ = range(-int(filter_size/2),int(filter_size/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in range_]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cartprod(*arrays):
     N = len(arrays)
     return np.transpose(np.meshgrid(*arrays, indexing='ij'),
                      np.roll(np.arange(N + 1), -1)).reshape(-1, N)

def combinations_without_repeat(a):
    n = len(a)
    out = np.empty((n,n-1,2),dtype=a.dtype)
    out[:,:,0] = np.broadcast_to(a[:,None], (n, n-1))
    out.shape = (n-1,n,2)
    out[:,:,1] = onecold(a)
    out.shape = (-1,2)
    return out

def onecold(a):
    n = len(a)
    s = a.strides[0]
    strided = np.lib.stride_tricks.as_strided
    b = np.concatenate((a,a[:-1]))
    return strided(b[1:], shape=(n-1,n), strides=(s,s))

def bytes_from_file(filename, chunksize=8192):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def draw_angled_rec(x0, y0, width, height, angle, img):

    _angle = angle * np.pi / 180.0
    b = np.cos(_angle) * 0.5
    a = np.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, (0, 0, 255), 3)
    cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    cv2.line(img, pt2, pt3, (255, 255, 255), 3)
    cv2.line(img, pt3, pt0, (255, 255, 255), 3)

def get_keypts(filename):
    global resized_keypts
    if filename:
        with open(filename, 'r') as f:
            datastore = json.load(f)

    if len(datastore['people']) > 0:
        jsonnn_tree = objectpath.Tree(datastore['people'][0]) # we only take the first person / skeleton 0 for our task
        result_tuple = tuple(jsonnn_tree.execute('$..pose_keypoints_2d'))
        key_pts = np.array(result_tuple)
        previous_keypts = resized_keypts.copy()
        resized_keypts = np.resize(key_pts, (25, 3)) # 25 2D keypoints, 3 values, x, y and confidence

def fill_or_shift(image):
    global fill_list, image_vector, key_pts_vector
    if fill_list == True:
        image_vector.append(image.copy())
        image_vector = np.repeat(image_vector, record_num_of_frames, axis=0)
        # print("size of image vector: ", np.size(image_vector, 0))
        key_pts_vector.append(resized_keypts.copy())
        key_pts_vector = np.repeat(key_pts_vector, record_num_of_frames, axis=0)
        # print("size of key_pts_vector: ", np.size(key_pts_vector, 0))
        fill_list = False;
    else:
        image_vector = np.roll(image_vector, -np.size(image))    # shift to left, by the size of one image
        image_vector[record_num_of_frames-1] = image.copy()     # replace right most image with a new copy

        key_pts_vector = np.roll(key_pts_vector, -1, axis=0)
        key_pts_vector[record_num_of_frames-1] = resized_keypts.copy()     # replace right most image with a new copy

def repeat_missing_pts(resized_keypts):
    global previous_val_check, key_pts_vector
    for l in range(0, np.size(resized_keypts, 0)):
        if previous_val_check[l] < track_missed_pt_num_of_prev_frames:
            # output_string = str(key_pts_vector[:,l,0]) + " " + str(key_pts_vector[record_num_of_frames-1,l,0]) \
            #     + " -- " + str(str(key_pts_vector[:,l,1])) + " " + str(key_pts_vector[record_num_of_frames-1,l,1])
            # print(output_string)
            if key_pts_vector[record_num_of_frames-1,l,0] == 0 and key_pts_vector[record_num_of_frames-1,l,1] == 0 \
                and key_pts_vector[record_num_of_frames-2,l,0] != 0 and key_pts_vector[record_num_of_frames-2,l,1] != 0:
                key_pts_vector[record_num_of_frames-1,l,0] = key_pts_vector[record_num_of_frames-2,l,0]
                key_pts_vector[record_num_of_frames-1,l,1] = key_pts_vector[record_num_of_frames-2,l,1]
                previous_val_check[l] += 1
                zero_str = "Last Frame Zero Found at joint: " + joint_names[l] + ". Replaced with the previous non-zero value!"
                print(zero_str)
                zero_str = "Previous val check of joint: " + joint_names[l] + " is : " + str(previous_val_check[l])
                print(zero_str)
            else:
                previous_val_check[l] = 0   # if no value is missed, reset the counter
            if key_pts_vector[record_num_of_frames-1,l,0] != 0 and key_pts_vector[record_num_of_frames-1,l,1] != 0\
                and key_pts_vector[record_num_of_frames-2,l,0] == 0 and key_pts_vector[record_num_of_frames-2,l,1] == 0:
                for yy in range(0, record_num_of_frames-1):
                    key_pts_vector[yy, l, 0] = key_pts_vector[record_num_of_frames-1,l,0]
                    key_pts_vector[yy, l, 1] = key_pts_vector[record_num_of_frames-1,l,1]
                zero_str = "Non-zero value appeared at: " + str(joint_names[l]) + ". Replaced previous value in the list with this non-zero value!"
                print(zero_str)
        else:
            zero_str = "Previous Value Check limit of joint " + joint_names[l] + " has reached."
            print(zero_str)
            zero_str = "Zeroing all previous " + str(record_num_of_frames) + " frames to prevent wrong smoothing by Gaussian filter."
            print(zero_str)

            # for yy in range(0, record_num_of_frames-1):
            key_pts_vector[:, l, 0] = 0
            key_pts_vector[:, l, 1] = 0
            previous_val_check[l] = 0   # if value check counter increases by
                                            # track_missed_pt_num_of_prev_frames,
                                            # no forward propagation of previous
                                            # value occurs, thus reset the counter.

def draw_skeleton(display_image, mid_array_keypoints):
    for m in range(0, np.size(mid_array_keypoints, 0) - 15): # Only top 14 keypoints are drawn
        if m == 8 or m == 9:
            continue
        # cv2.circle(output_image,(int(mid_array_keypoints[i,0]), int(mid_array_keypoints[i,1])), 3, (0,0,255), -1)
        if np.count_nonzero(mid_array_keypoints[m]):
            cv2.circle(display_image,(int(mid_array_keypoints[m,0]), int(mid_array_keypoints[m,1])), 3, (0,0,255), -1)

    for m in range(0, np.size(line_indices_1, 0)):
        # temp_1 = np.array((int(mid_array_keypoints[line_indices_1[m],0]), int(mid_array_keypoints[line_indices_1[m],1])))
        # temp_2 = np.array((int(mid_array_keypoints[line_indices_2[m],0]), int(mid_array_keypoints[line_indices_2[m],1])))
        temp_1 = np.round(mid_array_keypoints[line_indices_1[m],:2])
        temp_2 = np.round(mid_array_keypoints[line_indices_2[m],:2])
        if (np.count_nonzero(temp_1) and np.count_nonzero(temp_2)): # To avoid drawing lines if one of the point is (0, 0)
            # cv2.line(output_image, (temp_1[0], temp_1[1]), (temp_2[0], temp_2[1]), (0,0,255), 2, -1)
            cv2.line(display_image, (int(temp_1[0]), int(temp_1[1])), (int(temp_2[0]), int(temp_2[1])), (0,0,255), 2, -1)

def get_selected_joints(joints_selected):
    pts = []    # Because we only select non-zero points
    missing_pts = []
    missing_pts = np.ones((np.size(joints_selected), 1))
    for m in range(0, np.size(joints_selected)):   # Until right wrist + left_shoulder
        if np.count_nonzero(mid_array_keypoints[joints_selected[m],:2]):
            if m == 0:
                pts =  mid_array_keypoints[joints_selected[m],:2]
            elif np.count_nonzero(pts):
                pts = np.vstack([pts, mid_array_keypoints[joints_selected[m],:2]])
        else:
            missing_pts[m] = 0
    return pts, missing_pts

def fit_line(pts, *args, **kwargs):
    display_image = kwargs.get('display_image', None)
    show = kwargs.get('show', None)
    [vx,vy,x,y] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

    # <<< To estimate a line fit on upper_right_pts + left_shoulder
    if show:
        cv2.circle(display_image,(x, y), 3, (255,255,0), -1)    # Center of Mass (of non-zero selected points)
        # >>> Only to display the line from two extremes of image
        # Now find two extreme points on the line to draw line
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((display_image.shape[1]-x)*vy/vx)+y)

        #Finally draw the line
        # cv2.line(display_image,(display_image.shape[1]-1,righty),(0,lefty),255,2)

        side_1_x = x + vx * 100
        side_1_y = y + vy * 100

        side_2_x = x - vx * 100
        side_2_y = y - vy * 100

        cv2.line(display_image,(x, y),(side_1_x,side_1_y),255,2)    # x is column, y is row
        cv2.line(display_image,(x, y),(side_2_x,side_2_y),255,2)
    return vx, vy, x, y

def get_mean_depth(joint_index, keypoints, average_grid_size):
    all_depth = np.zeros((average_grid_size, average_grid_size))
    start_index = (average_grid_size - 1) / 2

    if np.count_nonzero(keypoints[joint_index,:2]):
        # print("Point is: ", mid_array_keypoints[4,:2])
        for m in range(0, average_grid_size):
            for n in range(0, average_grid_size):
                row_index = int(keypoints[joint_index, 1] - start_index + m)
                if row_index < 0:
                    row_index = 0
                if row_index > np.size(image_vector[display_frame], 0) - 1:
                    row_index = np.size(image_vector[display_frame], 0) - 1
                col_index = int(keypoints[joint_index, 0] - start_index + n)
                if col_index < 0:
                    col_index = 0
                if col_index > np.size(image_vector[display_frame], 1) - 1:
                    col_index = np.size(image_vector[display_frame], 1) - 1
                all_depth[m, n] = np.mean(image_vector[display_frame][row_index, col_index])
        flatten_depth = all_depth.flatten()
        idx = [idx for idx, el in enumerate(flatten_depth) if el < 1]
        flatten_depth = np.delete(flatten_depth, idx)
        mean_d = np.mean(flatten_depth)
        # print("Flatten depth", flatten_depth)
    else:
        print ("Right Wrist not found!")
        mean_d = 0
    return mean_d

def get_link_distances(joints_selected, missing):
    zero_vec_check_list = joints_selected
    zero_check_combinations = sorted(set(combinations(zero_vec_check_list, 2)), reverse=False)
    zero_check = np.array(zero_check_combinations).tolist()

    # combinations_list_temp = np.arange(1, np.size(joints_selected) + 1)
    combinations_list_temp = [i + 1 for i in joints_selected]
    combinations_list_temp = np.multiply(combinations_list_temp, missing.T)
    combinations_list = [int(combinations_list - 1) for combinations_list in combinations_list_temp[0] if combinations_list != 0]
    combinations_idx = sorted(set(combinations(combinations_list, 2)), reverse=True)
    # print(combinations_idx)

    angle_combination_list = np.arange(len(combinations_idx))
    angle_combinations = sorted(set(combinations(angle_combination_list, 2)), reverse=False)

    vec = np.zeros((len(zero_check_combinations), 2))
    link_distances = np.zeros((np.size(vec, 0), 1))

    for m in range(0, len(combinations_idx)):
        elem = np.array(combinations_idx.pop())
        idx = [idx for idx, el in enumerate(zero_check) if np.array_equal(el, elem)]
        vec[idx] = np.round([mid_array_keypoints[elem[0],:2] - mid_array_keypoints[elem[1],:2]])
        link_distances[idx] = la.norm(vec[idx])

    return vec, link_distances

def apply_gaussian(key_points_vector):
    Gauss_vals = Gaussian_filter(filter_size = record_num_of_frames, sigma = 1)
    # print(np.shape(key_points_vector))
    Gauss_vals = np.repeat(Gauss_vals, np.size(key_points_vector[0]), axis=0)
    Gauss_vals = np.reshape(Gauss_vals, np.shape(key_points_vector))
    Gauss_filtered_middle = np.sum(np.multiply(key_points_vector, Gauss_vals), axis=0)
    # print(Gauss_filtered_middle)
    # c = np.multiply(a, b)
    # print(c)

    return Gauss_filtered_middle.copy()

def get_angles(vec_index, vec, vx, vy, part):
    if part == 0 or part == 1:
        angles = np.zeros((len(vec_index)*2 - 1, 1))
        for m in np.arange(len(vec_index) - 1):
            if m < len(vec_index) - 2:
                if np.count_nonzero(vec[vec_index[m]]) and np.count_nonzero(vec[vec_index[m+1]]):
                    angles[m] = angle_between(vec[vec_index[m]], -vec[vec_index[m+1]])
            else:
                if np.count_nonzero(vec[vec_index[0]]) and np.count_nonzero(vec[vec_index[m+1]]):
                    angles[m] = angle_between(vec[vec_index[0]], -vec[vec_index[m+1]])

        for m in range(4, np.size(angles, 0)):
            if np.count_nonzero(vec[vec_index[m-4]]) and np.count_nonzero([vx,vy]):
                angles[m] = angle_between(vec[vec_index[m-4]], [vx,vy])
    if part == 2:
        angles = np.zeros((len(vec_index)*2 - 1, 1))    # OK
        for m in np.arange(len(vec_index) - 1):
            if m < 3:
                if np.count_nonzero(vec[vec_index[m]]) and np.count_nonzero(vec[vec_index[m+1]]):
                    angles[m] = angle_between(vec[vec_index[m]], -vec[vec_index[m+1]])
            elif m == 3:
                if np.count_nonzero(vec[vec_index[0]]) and np.count_nonzero(vec[vec_index[m+1]]):
                    angles[m] = angle_between(vec[vec_index[0]], -vec[vec_index[m+1]])
            elif m > 3:
                if np.count_nonzero(vec[vec_index[m]]) and np.count_nonzero(vec[vec_index[m+1]]):
                    angles[m] = angle_between(vec[vec_index[m]], -vec[vec_index[m+1]])
            for m in range(6, np.size(angles, 0)):
                if np.count_nonzero(vec[vec_index[m-6]]) and np.count_nonzero([vx,vy]):
                    angles[m] = angle_between(vec[vec_index[m-6]], [vx,vy])
    return angles

def get_feature_vector(vec, link_distances, angles):
    descriptors = []
    moved_coordinates = vec.flatten('F')
    moved_coordinates = np.asarray(moved_coordinates).reshape(np.size(vec), 1)
    descriptors = np.append(descriptors, moved_coordinates)
    descriptors = np.append(descriptors, link_distances)
    descriptors = np.append(descriptors, angles * 180 / np.pi)

    descriptors = np.asarray(descriptors).reshape(np.size(descriptors), 1)

    return descriptors

def create_feature_vector(final_descriptors, mean_d_wrist, vec, mid_array_keypoints, display_image, coordinate_idx_number, *args, **kwargs):
    mean_d_elbow = kwargs.get('mean_d_elbow', None)
    mean_d_shoulder = kwargs.get('mean_d_shoulder', None)
    global right_feature_vector, right_mean_depth_vector, right_rec_size_vector, \
    left_feature_vector, left_mean_depth_vector, left_rec_size_vector, \
    neck_feature_vector, neck_mean_depth_vector, neck_rec_size_vector, \
    right_elbow_depth_vector, right_shoulder_depth_vector, \
    left_elbow_depth_vector, left_shoulder_depth_vector

    if mean_d_wrist != 0 and not np.isnan(mean_d_wrist) and not np.isnan(vec[coordinate_idx_number,[0]]) \
        and not np.isnan(vec[coordinate_idx_number,[1]]) and vec[coordinate_idx_number,[0]] != 0 \
        and vec[coordinate_idx_number,[1]] != 0 and mean_d_wrist != np.inf:    # right_vec[12] this index (12th) comes from combinations_idx for right side.
                                                                # this mean length of right-forearm
                                                                # for left side, it will be 14th (from left combination_idx)
        bbox_tuning = 550
        # box_center = mid_array_keypoints[4,:2] + link_distances[12] / 2
        if coordinate_idx_number == 12:
            box_center = mid_array_keypoints[4,:2] - ((vec[coordinate_idx_number] / la.norm(vec[coordinate_idx_number])) * right_link_distances[coordinate_idx_number] / 2)
            rec_scale = bbox_tuning / mean_d_wrist * 500
            # rec_scale = bbox_tuning / y_right[0][0] * 500
            angle = atan2(mid_array_keypoints[4, 1] - mid_array_keypoints[3, 1], mid_array_keypoints[4, 0] - mid_array_keypoints[3, 0]) * 180.0 / np.pi;
            # For description
            # angle = atan2(wrist_joint.y - elbow_joint.y, wrist_joint.x - elbow_joint.x) * 180.0 / np.pi;
            if len(right_feature_vector) == 0:
                right_feature_vector = final_descriptors
            else:
                right_feature_vector = np.concatenate((right_feature_vector, final_descriptors), axis = 1)
            if np.shape(right_mean_depth_vector) == 0:
                right_mean_depth_vector = mean_d_wrist
            else:
                right_mean_depth_vector = np.append(right_mean_depth_vector, mean_d_wrist)
            if np.shape(right_elbow_depth_vector) == 0:
                right_elbow_depth_vector = mean_d_elbow
            else:
                right_elbow_depth_vector = np.append(right_elbow_depth_vector, mean_d_elbow)
            if np.shape(right_shoulder_depth_vector) == 0:
                right_shoulder_depth_vector = mean_d_shoulder
            else:
                right_shoulder_depth_vector = np.append(right_shoulder_depth_vector, mean_d_shoulder)
            if np.shape(right_rec_size_vector) == 0:
                right_rec_size_vector = rec_scale
            else:
                right_rec_size_vector = np.append(right_rec_size_vector, rec_scale)
            draw_angled_rec(int(box_center[0]), int(box_center[1]), rec_scale, rec_scale, angle, display_image)

        elif coordinate_idx_number == 14:
            box_center = mid_array_keypoints[7,:2] - ((vec[coordinate_idx_number] / la.norm(vec[coordinate_idx_number])) * left_link_distances[coordinate_idx_number] / 2)
            rec_scale = bbox_tuning / mean_d_wrist * 500
            # rec_scale = bbox_tuning / y_left[0][0] * 500
            angle = atan2(mid_array_keypoints[7, 1] - mid_array_keypoints[6, 1], mid_array_keypoints[7, 0] - mid_array_keypoints[6, 0]) * 180.0 / np.pi;
            if len(left_feature_vector) == 0:
                left_feature_vector = final_descriptors
            else:
                left_feature_vector = np.concatenate((left_feature_vector, final_descriptors), axis = 1)
            if np.shape(left_mean_depth_vector) == 0:
                left_mean_depth_vector = mean_d_wrist
            else:
                left_mean_depth_vector = np.append(left_mean_depth_vector, mean_d_wrist)
            if np.shape(left_elbow_depth_vector) == 0:
                left_elbow_depth_vector = mean_d_elbow
            else:
                left_elbow_depth_vector = np.append(left_elbow_depth_vector, mean_d_elbow)
            if np.shape(left_shoulder_depth_vector) == 0:
                left_shoulder_depth_vector = mean_d_shoulder
            else:
                left_shoulder_depth_vector = np.append(left_shoulder_depth_vector, mean_d_shoulder)
            if np.shape(left_rec_size_vector) == 0:
                left_rec_size_vector = rec_scale
            else:
                left_rec_size_vector = np.append(left_rec_size_vector, rec_scale)
            draw_angled_rec(int(box_center[0]), int(box_center[1]), rec_scale, rec_scale, angle, display_image)
        elif coordinate_idx_number == 0:
            box_center = mid_array_keypoints[1,:2]
            rec_scale = bbox_tuning / mean_d_wrist * 100
            # rec_scale = bbox_tuning / y_left[0][0] * 500
            angle = atan2(mid_array_keypoints[1, 1] - mid_array_keypoints[0, 1], mid_array_keypoints[1, 0] - mid_array_keypoints[0, 0]) * 180.0 / np.pi;
            if len(neck_feature_vector) == 0:
                neck_feature_vector = final_descriptors
            else:
                neck_feature_vector = np.concatenate((neck_feature_vector, final_descriptors), axis = 1)
            if np.shape(neck_mean_depth_vector) == 0:
                neck_mean_depth_vector = mean_d_wrist
            else:
                neck_mean_depth_vector = np.append(neck_mean_depth_vector, mean_d_wrist)
            if np.shape(neck_rec_size_vector) == 0:
                neck_rec_size_vector = rec_scale
            else:
                neck_rec_size_vector = np.append(neck_rec_size_vector, rec_scale)


joint_names = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", \
                "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", \
                "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", \
                "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", \
                "RHeel", "Background"]

depth_folder_names = ["Anthony", "Arnaud", "Arthur", "Benjamin",\
                        "Francois", "Jihong1_d", "Kevin", "Stephane", "Zineb"]
                        # arranged with respect to JSON files folder

                                #     {0,  "Nose"},
                                #     {1,  "Neck"},
                                #     {2,  "RShoulder"},
                                #     {3,  "RElbow"},
                                #     {4,  "RWrist"},
                                #     {5,  "LShoulder"},
                                #     {6,  "LElbow"},
                                #     {7,  "LWrist"},
                                #     {8,  "MidHip"},
                                #     {9,  "RHip"},
                                #     {10, "RKnee"},
                                #     {11, "RAnkle"},
                                #     {12, "LHip"},


# # >>> Get Individual Json files for each video in each folder/subfolders
json_folder = "/media/osama/Elements/dataset/JSON_files/*"

line_indices_1 = [1, 1, 1, 2, 3, 5, 6]
line_indices_2 = [0, 2, 5, 3, 4, 6, 7]

resized_keypts = np.zeros((25,3))               # actually to save/initialze previous_keypts in the first frame
previous_val_check = np.zeros((25,1))           # To monitor how many times each point is missed in repetition
record_num_of_frames = 7                        # This MUST be an odd value
track_missed_pt_num_of_prev_frames = record_num_of_frames  # We forward propagate the previous known value of a missed point for this much of times/frames

right_feature_vector = []
right_mean_depth_vector = []
right_rec_size_vector = []
right_elbow_depth_vector = []
right_shoulder_depth_vector = []

left_feature_vector = []
left_mean_depth_vector = []
left_rec_size_vector = []
left_elbow_depth_vector = []
left_shoulder_depth_vector = []

neck_feature_vector = []
neck_mean_depth_vector = []
neck_rec_size_vector = []

image_vector = []
key_pts_vector = []
fill_list = True

for x in range(0, len(depth_folder_names)):
    depth_video_folder_name = "/media/osama/Elements1/GestureRecordings/" + depth_folder_names[x] + "/*.bin"
    current_folder = depth_folder_names[x]
    print(depth_video_folder_name)
    # depth_image_files = [os.path.basename(x) for x in glob(depth_video_folder_name)]
    depth_image_files = glob(depth_video_folder_name)
    depth_image_files.sort(key=natural_keys)

    json_root_folder_paths = glob(json_folder)
    # json_root_folder_paths.sort()    # glob gives a random order so we sort the list of folders here

    json_root = json_root_folder_paths[x] + '/*'    # Names of volunteers
    print("json_root", json_root)

    json_gestures_paths = glob(json_root)
    json_gestures_paths.sort()

    depth_image = np.zeros((1082, 1920), dtype=np.float)
    flat_image = np.zeros((1, 1082* 1920), dtype = np.float)

    image_vector = []
    key_pts_vector = []
    fill_list = True    # To fill image_vector for the first time only
    display_frame =  int((record_num_of_frames - 1) / 2)

    list = np.arange(1, 10)
    # starting_image_index = len(json_gestures_paths) - 11 # constant cannot be less than 11
    starting_image_index = 0

    for i in range(starting_image_index, len(json_gestures_paths)):
        xyscalar, linkscalar, angle_scalar, Yscalar = MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
        start = time.time()
        temp_fill = []
        temp_fill = array.array("f") # B is the typecode for uchar
        print("json_root: ", json_gestures_paths[i])
        print("depth file: ", depth_image_files[i])
        if current_folder in ["Jihong1_d"]:
            # # >>> bin with only depth image
            with open(depth_image_files[i], 'rb') as fid:
                fid.seek(8)   # We skip these bytes (2 ints (or 8 bytes) for size of image and 1080*1920*4 (uchar) bytes for rgb image)
                temp_fill.fromfile(fid, 1082*1920)
                flat_image = np.asarray(temp_fill.tolist())
                depth_image = np.resize(flat_image, (1082, 1920))
                end = time.time()
                print(str("Time taken to load the depth image: " + str(round((end - start), 2)) + " secs"))
            # # <<< bin with only depth image
        else:
            # # >> START HERE BIN WITH BOTH RGB AND DEPTH
            with open(depth_image_files[i], 'rb') as fid:
                # a = array.array("i")  # i is the typecode for signed int
                # a.fromfile(fid, 2)
                # print(a.tolist())
                # print(a)
                fid.seek(8 + 1080*1920*4)   # We skip these bytes (2 ints (or 8 bytes) for size of image and 1080*1920*4 (uchar) bytes for rgb image)
                # b = array.array("B") # B is the typecode for uchar
                # b.fromfile(fid, 1080*1920*4)
                # b = np.asarray(b.tolist())
                # b = np.resize(b, (1080, 1920, 4))

                temp_fill.fromfile(fid, 1082*1920)
                flat_image = np.asarray(temp_fill.tolist())
                depth_image = np.resize(flat_image, (1082, 1920))
                end = time.time()
                print(str("Time taken to load the depth image: " + str(round((end - start), 2)) + " secs"))
                # cv2.imshow("test image", b)
        fid.close()
        # # << START HERE START HERE BIN WITH BOTH RGB AND DEPTH

        filename = json_gestures_paths[i]
        get_keypts(filename)
        fill_or_shift(depth_image)
        repeat_missing_pts(resized_keypts)

        if i >= int(record_num_of_frames - 1):   # -1 as count starts from 0
            mid_array_keypoints = apply_gaussian(key_pts_vector)

            col_max, row_max, _ = np.max(mid_array_keypoints, axis=0) # max of each column
            # print(col_max, row_max) # for determining the size of output image
            output_image = np.zeros((int(row_max + 40), int(col_max + 40), 3), dtype = 'uint8')
            # Make display image a 3channel image
            # print("Size of one image: ", str(np.size(image_vector[display_frame], 0)) + ", " + str(np.size(image_vector[display_frame], 1)))
            temp_image = image_vector[display_frame].copy()
            display_image = np.stack((temp_image / 4096,)*3, axis=-1)
            # print("Size of display image: ", np.shape(display_image))
            # Result for BODY_25 (25 body parts consisting of COCO + foot)
            # const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
            #     {0,  "Nose"},
            #     {1,  "Neck"},
            #     {2,  "RShoulder"},
            #     {3,  "RElbow"},
            #     {4,  "RWrist"},
            #     {5,  "LShoulder"},
            #     {6,  "LElbow"},
            #     {7,  "LWrist"},
            #     {8,  "MidHip"},
            #     {9,  "RHip"},
            #     {10, "RKnee"},
            #     {11, "RAnkle"},
            #     {12, "LHip"},
            #     {13, "LKnee"},
            #     {14, "LAnkle"},
            #     {15, "REye"},
            #     {16, "LEye"},
            #     {17, "REar"},
            #     {18, "LEar"},
            #     {19, "LBigToe"},
            #     {20, "LSmallToe"},
            #     {21, "LHeel"},
            #     {22, "RBigToe"},
            #     {23, "RSmallToe"},
            #     {24, "RHeel"},
            #     {25, "Background"}
            # };

            draw_skeleton(display_image, mid_array_keypoints)
            top_right_joints_selected = [0, 1, 2, 3, 4, 5]  # openpose joint index
            top_left_joints_selected  = [0, 1, 2, 5, 6, 7]
            top_joints_selected  = [0, 1, 2, 3, 4, 5, 6, 7]

            upper_right_pts, missing_upper_right = get_selected_joints(top_right_joints_selected)
            upper_left_pts,  missing_upper_left  = get_selected_joints(top_left_joints_selected )
            upper_pts,  missing_upper  = get_selected_joints(top_joints_selected )

            # then apply fitline() function
            average_grid_size = 5   # try to keep it odd
            if np.count_nonzero(upper_right_pts):
                r_vx, r_vy, r_x, r_y = fit_line(upper_right_pts, display_image=display_image, show=True)
                right_shoulder_mean_depth  = get_mean_depth(2, mid_array_keypoints, average_grid_size)
                right_elbow_mean_depth  = get_mean_depth(3, mid_array_keypoints, average_grid_size)
                right_wrist_mean_depth = get_mean_depth(4, mid_array_keypoints, average_grid_size)
                right_vec, right_link_distances = get_link_distances(top_right_joints_selected, missing_upper_right)
                right_vec_index = [0, 5, 9, 12, 8 ] # get it by printing combination_idx # vector index # P2,P1 # P2 - P1
                right_side_angles = get_angles(right_vec_index, right_vec, r_vx, r_vy, 0)
                right_final_descriptors = get_feature_vector(right_vec, right_link_distances, right_side_angles)

                create_feature_vector(right_final_descriptors, right_wrist_mean_depth, \
                right_vec, mid_array_keypoints, display_image, 12, \
                mean_d_elbow=right_elbow_mean_depth, mean_d_shoulder=right_shoulder_mean_depth)

            if np.count_nonzero(upper_left_pts):
                l_vx, l_vy, l_x, l_y = fit_line(upper_left_pts, display_image=display_image, show=True)
                left_shoulder_mean_depth  = get_mean_depth(5, mid_array_keypoints, average_grid_size)
                left_elbow_mean_depth  = get_mean_depth(6, mid_array_keypoints, average_grid_size)
                left_wrist_mean_depth  = get_mean_depth(7, mid_array_keypoints, average_grid_size)
                left_vec,  left_link_distances  = get_link_distances(top_left_joints_selected,  missing_upper_left )
                left_vec_index =  [0, 5, 6, 12, 14]
                left_side_angles  = get_angles(left_vec_index,  left_vec,  l_vx, l_vy, 1)
                left_final_descriptors  = get_feature_vector(left_vec, left_link_distances, left_side_angles)

                create_feature_vector(left_final_descriptors, left_wrist_mean_depth, \
                left_vec, mid_array_keypoints, display_image, 14, \
                mean_d_elbow=left_elbow_mean_depth, mean_d_shoulder=left_shoulder_mean_depth)

            if np.count_nonzero(upper_pts):
                u_vx, u_vy, u_x, u_y = fit_line(upper_pts, display_image=display_image, show=True)
                neck_mean_depth  = get_mean_depth(1, mid_array_keypoints, average_grid_size)
                upper_vec,  upper_link_distances  = get_link_distances(top_joints_selected,  missing_upper )
                upper_vec_index = [0, 7, 13, 18, 10, 25, 27] 
                upper_angles      = get_angles(upper_vec_index, upper_vec, u_vx, u_vy, 2)
                upper_final_descriptors = get_feature_vector(upper_vec, upper_link_distances, upper_angles)

                create_feature_vector(upper_final_descriptors, neck_mean_depth, \
                                        upper_vec, mid_array_keypoints, display_image, 0)


            # To do: remove the vector creation code and replace it with the indices obtained from elem. Also automate angle estimation for all vectors obtained.
            # >>> To show all images
            # for ii in range(0, record_num_of_frames):
            #     window_title = "Video Output" + str(ii)
            #     cv2.namedWindow(window_title)
            #     if ii != display_frame:
            #         cv2.imshow(window_title, image_vector[ii])
            #     else:
            #         cv2.imshow(window_title, display_image)
            # <<< To show all images

            cv2.namedWindow("Main video output", cv2.WINDOW_NORMAL)
            cv2.imshow('Main video output', np.array(display_image, dtype = np.float ))
            k = cv2.waitKey(1)
            if k==27:    # Esc key to stop
                cv2.destroyAllWindows()
                print("Escape pressed... Program Closing!")
                exit()
print('-------------------------------------')
print('Writing Right Feature Vector:        ', np.shape(right_feature_vector))
print('Writing Right Mean Depth Vector:     ', np.shape(right_mean_depth_vector))
print('Writing Right Elbow Depth Vector:    ', np.shape(right_elbow_depth_vector))
print('Writing Right Shoulder Depth Vector: ', np.shape(right_shoulder_depth_vector))
print('Writing Right Rec Scale Vector:      ', np.shape(right_rec_size_vector))
print('-------------------------------------')
print('Writing Left  Mean Depth Vector:     ', np.shape(left_mean_depth_vector))
print('Writing Left  Elbow Depth Vector:    ', np.shape(left_elbow_depth_vector))
print('Writing Left  Feature Vector:        ', np.shape(left_feature_vector))
print('Writing Left  Shoulder Depth Vector: ', np.shape(left_shoulder_depth_vector))
print('Writing Left  Rec Scale Vector:      ', np.shape(left_rec_size_vector))
print('-------------------------------------')
print('Writing Neck  Feature Vector:        ', np.shape(neck_feature_vector))
print('Writing Neck  Mean Depth Vector:     ', np.shape(neck_mean_depth_vector))
print('Writing Neck  Rec Scale Vector:      ', np.shape(neck_rec_size_vector))
print('-------------------------------------')
print("Writing data into h5 file")
print('-------------------------------------')
file_h5 = h5py.File('all_upper_kinect2_data.h5', 'w')
file_h5.create_dataset('right_feature_vector', data = right_feature_vector)
file_h5.create_dataset('right_mean_depth_vector', data = right_mean_depth_vector)
file_h5.create_dataset('right_elbow_depth_vector', data = right_elbow_depth_vector)
file_h5.create_dataset('right_shoulder_depth_vector', data = right_shoulder_depth_vector)
file_h5.create_dataset('right_rec_size_vector', data = right_rec_size_vector)

file_h5.create_dataset('left_feature_vector', data = left_feature_vector)
file_h5.create_dataset('left_mean_depth_vector', data = left_mean_depth_vector)
file_h5.create_dataset('left_elbow_depth_vector', data = left_elbow_depth_vector)
file_h5.create_dataset('left_shoulder_depth_vector', data = left_shoulder_depth_vector)
file_h5.create_dataset('left_rec_size_vector', data = left_rec_size_vector)

file_h5.create_dataset('neck_feature_vector', data = neck_feature_vector)
file_h5.create_dataset('neck_mean_depth_vector', data = neck_mean_depth_vector)
file_h5.create_dataset('neck_rec_size_vector', data = neck_rec_size_vector)
file_h5.close()
print('---------------------------------')
print("Data written")
print('---------------------------------')
#
#
# For reading h5py file
print("Reading data into h5 file")
print('---------------------------------')
read_h5 = h5py.File('all_upper_kinect2_data.h5', 'r')
read_right_feature_vector = read_h5['right_feature_vector'][:]
read_right_mean_depth_vector = read_h5['right_mean_depth_vector'][:]
read_right_elbow_depth_vector = read_h5['right_elbow_depth_vector'][:]
read_right_shoulder_depth_vector = read_h5['right_shoulder_depth_vector'][:]
read_right_rec_size_vector = read_h5['right_rec_size_vector'][:]

read_left_feature_vector = read_h5['left_feature_vector'][:]
read_left_mean_depth_vector = read_h5['left_mean_depth_vector'][:]
read_left_rec_size_vector = read_h5['left_rec_size_vector'][:]
read_left_elbow_depth_vector = read_h5['left_elbow_depth_vector'][:]
read_left_shoulder_depth_vector = read_h5['left_shoulder_depth_vector'][:]

read_neck_feature_vector = read_h5['neck_feature_vector'][:]
read_neck_mean_depth_vector = read_h5['neck_mean_depth_vector'][:]
read_neck_rec_size_vector = read_h5['neck_rec_size_vector'][:]
read_h5.close()
print("Data read")
print('----------------------------------')
# np.allclose(total_images, read_train_images)
print('Read Right Feature Vector:        ', np.shape(read_right_feature_vector))
print('Read Right Mean Depth Vector:     ', np.shape(read_right_mean_depth_vector))
print('Read Right Elbow Depth Vector:    ', np.shape(read_right_elbow_depth_vector))
print('Read Right Shoulder Depth Vector: ', np.shape(read_right_shoulder_depth_vector))
print('Read Right Rec Scale Vector:      ', np.shape(read_right_rec_size_vector))
print('----------------------------------')
print('Read Left  Feature Vector:        ', np.shape(read_left_feature_vector))
print('Read Left  Mean Depth Vector:     ', np.shape(read_left_mean_depth_vector))
print('Read Left  Elbow Depth Vector:    ', np.shape(read_left_elbow_depth_vector))
print('Read Left  Shoulder Depth Vector: ', np.shape(read_left_shoulder_depth_vector))
print('Read Left  Rec Scale Vector:      ', np.shape(read_left_rec_size_vector))
print('----------------------------------')
print('Read Neck  Feature Vector:        ', np.shape(read_neck_feature_vector))
print('Read Neck  Mean Depth Vector:     ', np.shape(read_neck_mean_depth_vector))
print('Read Neck  Rec Scale Vector:      ', np.shape(read_neck_rec_size_vector))
print('----------------------------------')

cv2.destroyAllWindows()
