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
from keras.models import load_model
from keras import optimizers
from rnn_utils import *

from keras import backend as K
import tensorflow as tf

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

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def draw_angled_rec(x0, y0, width, height, angle, img, image_to_crop):
    _angle = angle * np.pi / 180.0
    b = np.cos(_angle) * 0.5
    a = np.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cnt = np.vstack((pt0, pt1, pt2, pt3))
    rect = np.zeros((4, 2), dtype = "float32")
    rect[0] = pt2   # we set the order of rectangle points here
    rect[1] = pt3
    rect[2] = pt0
    rect[3] = pt1
    # print("shape of cnt: {}".format(cnt.shape))
    # rect = cv2.minAreaRect(cnt)
    # print("rect: {}".format(rect))
    # # the order of the box points: bottom left, top left, top right,
    # # bottom right
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    #
    # print("bounding box: {}".format(box))
    # cv2.drawContours(img, [box], 0, (255, 255, 255), 2)
    cv2.line(img, pt0, pt1, (0, 0, 255), 3)
    cv2.line(img, pt1, pt2, (255, 255, 255), 3)
    cv2.line(img, pt2, pt3, (255, 255, 255), 3)
    cv2.line(img, pt3, pt0, (255, 255, 255), 3)

    # get width and height of the detected rectangle
    # width = int(rect[1][0])
    # height = int(rect[1][1])

    width = int(width)
    height = int(height)
    #
    # src_pts = box.astype("float32")
    # # corrdinate of the points in box points after the rectangle has been
    # # straightened

    dst = np.array([
		[0, 0],
		[width - 1, 0],
		[width - 1, height - 1],
		[0, height - 1]], dtype = "float32")

    # dst_pts = np.array([[0, height-1],
    #                     [0, 0],
    #                     [width-1, 0],
    #                     [width-1, height-1]], dtype="float32")
    #
    # # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    #
    # # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image_to_crop, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_resized = cv2.resize(warped,(hand_height, hand_width), interpolation=cv2.INTER_LINEAR)
    return warped_resized

def get_keypts(filename):
    global resized_keypts, lh_keypts, rh_keypts
    if filename:
        with open(filename, 'r') as f:
            datastore = json.load(f)

    if len(datastore['people']) > 0:
        jsonnn_tree = objectpath.Tree(datastore['people'][0]) # we only take the first person / skeleton 0 for our task
        pose_tuple = tuple(jsonnn_tree.execute('$..pose_keypoints_2d'))
        pose_key_pts = np.array(pose_tuple)
        resized_keypts = np.resize(pose_key_pts, (25, 3)) # 25 2D keypoints, 3 values, x, y and confidence

        lh_tuple = tuple(jsonnn_tree.execute('$..hand_left_keypoints_2d'))
        lh_key_pts = np.array(lh_tuple)
        lh_keypts = np.resize(lh_key_pts, (20, 3)) # 20 2D keypoints, 3 values, x, y and confidence

        rh_tuple = tuple(jsonnn_tree.execute('$..hand_right_keypoints_2d'))
        rh_key_pts = np.array(rh_tuple)
        rh_keypts = np.resize(rh_key_pts, (20, 3)) # 20 2D keypoints, 3 values, x, y and confidence

def fill_or_shift(image):
    global fill_list, image_vector, key_pts_vector, key_pts_vector_orig, lh_keypts_vector, rh_keypts_vector
    if fill_list == True:
        image_vector.append(image.copy())
        image_vector = np.repeat(image_vector, record_num_of_frames, axis=0)
        # print("size of image vector: ", np.size(image_vector, 0))
        key_pts_vector.append(resized_keypts.copy())
        key_pts_vector = np.repeat(key_pts_vector, record_num_of_frames, axis=0)

        key_pts_vector_orig.append(resized_keypts.copy())
        key_pts_vector_orig = np.repeat(key_pts_vector_orig, record_num_of_frames, axis=0)

        lh_keypts_vector.append(lh_keypts.copy())
        lh_keypts_vector = np.repeat(lh_keypts_vector, record_num_of_frames, axis=0)

        rh_keypts_vector.append(rh_keypts.copy())
        rh_keypts_vector = np.repeat(rh_keypts_vector, record_num_of_frames, axis=0)

        # print("size of key_pts_vector: ", np.size(key_pts_vector, 0))
        fill_list = False;
    else:
        image_vector = np.roll(image_vector, -np.size(image))    # shift to left, by the size of one image
        image_vector[record_num_of_frames-1] = image.copy()     # replace right most image with a new copy

        key_pts_vector = np.roll(key_pts_vector, -1, axis=0)
        key_pts_vector[record_num_of_frames-1] = resized_keypts.copy()     # replace right most image with a new copy

        key_pts_vector_orig = np.roll(key_pts_vector_orig, -1, axis=0)
        key_pts_vector_orig[record_num_of_frames-1] = resized_keypts.copy()     # replace right most image with a new copy

        lh_keypts_vector = np.roll(lh_keypts_vector, -1, axis=0)
        lh_keypts_vector[record_num_of_frames-1] = lh_keypts.copy()     # replace right most image with a new copy

        rh_keypts_vector = np.roll(rh_keypts_vector, -1, axis=0)
        rh_keypts_vector[record_num_of_frames-1] = rh_keypts.copy()     # replace right most image with a new copy

def repeat_missing_pts(resized_keypts, previous_val_check, key_pts_vector, *args, **kwargs):
    joint_names = kwargs.get('joint_names', None)
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
                if joint_names != None:
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
                if joint_names != None:
                    zero_str = "Non-zero value appeared at: " + str(joint_names[l]) + ". Replaced previous value in the list with this non-zero value!"
                    print(zero_str)
        else:
            if joint_names != None:
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
            cv2.circle(display_image,(int(mid_array_keypoints[m,0]), int(mid_array_keypoints[m,1])), 3, [0,0,255], -1)

    for m in range(0, np.size(line_indices_1, 0)):
        # temp_1 = np.array((int(mid_array_keypoints[line_indices_1[m],0]), int(mid_array_keypoints[line_indices_1[m],1])))
        # temp_2 = np.array((int(mid_array_keypoints[line_indices_2[m],0]), int(mid_array_keypoints[line_indices_2[m],1])))
        temp_1 = np.round(mid_array_keypoints[line_indices_1[m],:2])
        temp_2 = np.round(mid_array_keypoints[line_indices_2[m],:2])
        if (np.count_nonzero(temp_1) and np.count_nonzero(temp_2)): # To avoid drawing lines if one of the point is (0, 0)
            # cv2.line(output_image, (temp_1[0], temp_1[1]), (temp_2[0], temp_2[1]), (0,0,255), 2, -1)
            cv2.line(display_image, (int(temp_1[0]), int(temp_1[1])), (int(temp_2[0]), int(temp_2[1])), (0,0,255), 15, -1)
    # cv2.circle(display_image,(int(lh_keypts[0,0]), int(lh_keypts[0,1])), 10, (0,0,255), -1)

def get_selected_joints(joints_selected, array_keypoints):
    pts = []    # Because we only select non-zero points
    missing_pts = []
    missing_pts = np.ones((np.size(joints_selected), 1))
    for m in range(0, np.size(joints_selected)):   # Until right wrist + left_shoulder
        if np.count_nonzero(array_keypoints[joints_selected[m],:2]):
            if m == 0:
                pts =  array_keypoints[joints_selected[m],:2]
            elif np.count_nonzero(pts):
                pts = np.vstack([pts, array_keypoints[joints_selected[m],:2]])
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
        if joint_index == 4:
            print ("Right Wrist not found!")
        if joint_index == 7:
            print ("Left Wrist not found!")
        mean_d = 0
    return mean_d

def get_link_distances(joints_selected, missing, array_keypoints):
    zero_vec_check_list = joints_selected
    zero_check_combinations = sorted(set(combinations(zero_vec_check_list, 2)), reverse=False)
    zero_check = np.array(zero_check_combinations).tolist()

    # combinations_list_temp = np.arange(1, np.size(joints_selected) + 1)
    combinations_list_temp = [i + 1 for i in joints_selected]
    combinations_list_temp = np.multiply(combinations_list_temp, missing.T)
    combinations_list = [int(combinations_list - 1) for combinations_list in combinations_list_temp[0] if combinations_list != 0]
    combinations_idx = sorted(set(combinations(combinations_list, 2)), reverse=True)
    # print(missing)
    # print(combinations_idx)

    angle_combination_list = np.arange(len(combinations_idx))
    angle_combinations = sorted(set(combinations(angle_combination_list, 2)), reverse=False)

    vec = np.zeros((len(zero_check_combinations), 2))
    link_distances = np.zeros((np.size(vec, 0), 1))

    for m in range(0, len(combinations_idx)):
        elem = np.array(combinations_idx.pop())
        idx = [idx for idx, el in enumerate(zero_check) if np.array_equal(el, elem)]
        vec[idx] = np.round([array_keypoints[elem[0],:2] - array_keypoints[elem[1],:2]])
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

def get_feature_vector(vec, link_distances, angles, *args, **kwargs):
    velocities = kwargs.get('velocities', None)
    accelerations = kwargs.get('accelerations', None)

    descriptors = []
    moved_coordinates = vec.flatten('F')
    moved_coordinates = np.asarray(moved_coordinates).reshape(np.size(vec), 1)

    descriptors = np.append(descriptors, moved_coordinates)
    descriptors = np.append(descriptors, link_distances)
    descriptors = np.append(descriptors, angles * 180 / np.pi)

    if velocities is not None:
        flattened_velocities = velocities.flatten('F')
        flattened_velocities = np.asarray(flattened_velocities).reshape(np.size(flattened_velocities), 1)
        descriptors = np.append(descriptors, flattened_velocities)

    if accelerations is not None:
        flattened_accelerations = accelerations.flatten('F')
        flattened_accelerations = np.asarray(flattened_accelerations).reshape(np.size(flattened_accelerations), 1)
        descriptors = np.append(descriptors, flattened_accelerations)

    descriptors = np.asarray(descriptors).reshape(np.size(descriptors), 1)

    return descriptors

def create_feature_vector(final_descriptors, mean_d_wrist, vec, mid_array_keypoints, display_image, image_to_crop, coordinate_idx_number, *args, **kwargs):
    mean_d_elbow = kwargs.get('mean_d_elbow', None)
    mean_d_shoulder = kwargs.get('mean_d_shoulder', None)
    global right_feature_vector, right_mean_depth_vector, right_rec_size_vector, \
    left_feature_vector, left_mean_depth_vector, left_rec_size_vector, \
    neck_feature_vector, neck_mean_depth_vector, neck_rec_size_vector, \
    right_elbow_depth_vector, right_shoulder_depth_vector, \
    left_elbow_depth_vector, left_shoulder_depth_vector, cropped_right, \
    cropped_left, cropped_right_vector, cropped_left_vector

    if mean_d_wrist != 0 and not np.isnan(mean_d_wrist) and not np.isnan(vec[coordinate_idx_number,[0]]) \
        and not np.isnan(vec[coordinate_idx_number,[1]]) and vec[coordinate_idx_number,[0]] != 0 \
        and vec[coordinate_idx_number,[1]] != 0 and mean_d_wrist != np.inf:    # right_vec[12] this index (12th) comes from combinations_idx for right side.
                                                                # this mean length of right-forearm
                                                                # for left side, it will be 14th (from left combination_idx)
        bbox_tuning = 550
        if coordinate_idx_number == 12:
            box_center = right_hand_pos_mean[:2]
            # rec_scale = bbox_tuning / mean_d_wrist * 500  # If y_right prediction not present, use depth value from depth image
            if y_right[0][0] < Yscaler_right.data_min_:
                y_right[0][0] = Yscaler_right.data_min_
            rec_scale = bbox_tuning / y_right[0][0] * 500
            angle = atan2(mid_array_keypoints[4, 1] - mid_array_keypoints[3, 1], mid_array_keypoints[4, 0] - mid_array_keypoints[3, 0]) * 180.0 / np.pi;
            # For description
            # angle = atan2(wrist_joint.y - elbow_joint.y, wrist_joint.x - elbow_joint.x) * 180.0 / np.pi;
            if box_center.all() != 0:
                cropped_right = draw_angled_rec(int(box_center[0]), int(box_center[1]), rec_scale, rec_scale, angle, display_image, image_to_crop)
                cropped_right_vector.append(cropped_right)
            else:
                cropped_right_vector.append(np.zeros((int(hand_height), int(hand_width), 3), dtype = 'uint8'))

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

        elif coordinate_idx_number == 14:
            box_center = left_hand_pos_mean[:2]
            # rec_scale = bbox_tuning / mean_d_wrist * 500  # If y_left prediction not present, use depth value from depth image
            if y_left[0][0] < Yscaler_left.data_min_:
                y_left[0][0] = Yscaler_left.data_min_
            rec_scale = bbox_tuning / y_left[0][0] * 500
            angle = atan2(mid_array_keypoints[7, 1] - mid_array_keypoints[6, 1], mid_array_keypoints[7, 0] - mid_array_keypoints[6, 0]) * 180.0 / np.pi;
            if box_center.all() != 0:
                cropped_left = draw_angled_rec(int(box_center[0]), int(box_center[1]), rec_scale, rec_scale, angle, display_image, image_to_crop)
                cropped_left_vector.append(cropped_left)
            else:
                cropped_left_vector.append(np.zeros((int(hand_height), int(hand_width), 3), dtype = 'uint8'))

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

        elif coordinate_idx_number == 0:
            box_center = mid_array_keypoints[1,:2]
            # rec_scale = bbox_tuning / mean_d_wrist * 100  # If y_upper prediction not present, use depth value from depth image
            if y_upper[0][0] < Yscaler_neck.data_min_:
                y_upper[0][0] = Yscaler_neck.data_min_
            rec_scale = bbox_tuning / y_upper[0][0] * 500
            angle = atan2(mid_array_keypoints[1, 1] - mid_array_keypoints[0, 1], mid_array_keypoints[1, 0] - mid_array_keypoints[0, 0]) * 180.0 / np.pi;
            # draw_angled_rec(int(box_center[0]), int(box_center[1]), rec_scale, rec_scale, angle, display_image, image_to_crop)

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
    else:
        if coordinate_idx_number == 12:
            cropped_right_vector.append(np.zeros((int(hand_height), int(hand_width), 3), dtype = 'uint8'))
        if coordinate_idx_number == 14:
            cropped_left_vector.append(np.zeros((int(hand_height), int(hand_width), 3), dtype = 'uint8'))
        if coordinate_idx_number == 0:
            fill_empty = np.zeros((129,1))  # IMPORTANT - CHANGING FROM 97 to 129 due to acceleration and velocities
            if len(neck_feature_vector) == 0:
                neck_feature_vector = fill_empty
            else:
                neck_feature_vector = np.concatenate((neck_feature_vector, fill_empty), axis = 1)

def get_boudingbox(mid_array_keypoints, mid_array_h_keypoints, vec, link_distances, coordinate_idx_number):
    if coordinate_idx_number == 12:
        wrist_idx = 4
    elif coordinate_idx_number == 14:
        wrist_idx = 7

    for_hand_mean = mid_array_h_keypoints[np.all(mid_array_h_keypoints[:,:2] != 0, axis=1)]
    if for_hand_mean.size != 0:
        hand_pos_mean = np.mean(for_hand_mean[:,:2], axis = 0)
    else:
        if np.count_nonzero(mid_array_keypoints[wrist_idx,:2]):
            hand_pos_mean = mid_array_keypoints[wrist_idx,:2] - ((vec[coordinate_idx_number] / la.norm(vec[coordinate_idx_number])) * link_distances[coordinate_idx_number] / 2)
        else:
            hand_pos_mean = np.zeros((1,2))

    if hand_pos_mean.any() != 0 and not np.isnan(hand_pos_mean.any()):
        cv2.circle(display_image,(int(hand_pos_mean[0]), int(hand_pos_mean[1])), 10, (255,0,255), -1)
    return hand_pos_mean

def concatenate_images(image_vector, display_image, key_pts_vector_orig):
    init_output_width = 4480
    num_of_images_in_x = np.shape(image_vector)[0]
    center_image = int((num_of_images_in_x + 1) / 2)
    width_of_single_image = int(init_output_width / num_of_images_in_x)
    height_of_single_image = int(np.shape(image_vector)[1] * width_of_single_image / init_output_width)
    height_of_single_image = 480
    num_of_images_in_y = 1

    total_number_of_embedding_images = num_of_images_in_x * num_of_images_in_y

    frame_output_width = width_of_single_image * num_of_images_in_x
    frame_output_height = height_of_single_image * num_of_images_in_y
    image_box_line_thickness = 14
    count = 0
    output_image  = np.zeros((int(frame_output_height), int(frame_output_width), 3), dtype = 'uint8')
    for y in range(0, num_of_images_in_y):
        for x in range(0, num_of_images_in_x):
            # print(np.array2string(Y[:,count]))
            if x == center_image - 1:
                image = display_image
            else:
                image = image_vector[count,:,:,:].copy()
                draw_skeleton(image, key_pts_vector_orig[x])

            # image = image[:,:,::-1]
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
            if x == center_image - 1:
                cv2.rectangle(output_image, (x_position, int(y_position+(image_box_line_thickness/2))), \
                (int(x_position+width_of_single_image-(image_box_line_thickness/2)), \
                int(y_position+height_of_single_image-(image_box_line_thickness/2))), (255,0,0), image_box_line_thickness)
            count += 1
            cv2.waitKey(1)
    return output_image

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
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
K.set_session(sess)

pose_joint_names = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", \
    "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", \
    "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", \
    "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", \
    "RHeel", "Background"]

# Indices to draw lines between to related keypoints
# line_indices_1 = [0, 1, 1, 1, 2, 3, 5, 6, 8,  8,  9, 10, 12, 13]
# line_indices_2 = [1, 2, 5, 8, 3, 4, 6, 7, 9, 12, 10, 11, 13, 14]

line_indices_1 = [1, 1, 1, 2, 3, 5, 6]
line_indices_2 = [0, 2, 5, 3, 4, 6, 7]
hand_height = 224
hand_width = 224
resized_keypts = np.zeros((25,3))               # actually to save/initialze previous_keypts in the first frame
lh_keypts = np.zeros((20,3))                    # actually to save/initialze previous_keypts in the first frame
rh_keypts = np.zeros((20,3))                    # actually to save/initialze previous_keypts in the first frame
previous_pose_val_check = np.zeros((25,1))           # To monitor how many times each point is missed in repetition
previous_lh_val_check = np.zeros((20,1))           # To monitor how many times each point is missed in repetition
previous_rh_val_check = np.zeros((20,1))           # To monitor how many times each point is missed in repetition
record_num_of_frames = 7                        # This MUST be an odd value
track_missed_pt_num_of_prev_frames = record_num_of_frames  # We forward propagate the previous known value of a missed point for this much of times/frames
# json_folder = "/home/osama/Programs/SaveImagesfromKinect/Osama/JSON_files_flipped/*"
# rgb_video_path = "/home/osama/Programs/SaveImagesfromKinect/Osama/videos/rgb_video.avi"
model_right = load_model('./Trained_Models/right_hand-20190425121154/e-956-msle-0.000428.h5')
model_left = load_model('./Trained_Models/left_hand-20190425103611/e-995-msle-0.000450.h5')
model_neck = load_model('./Trained_Models/neck-20190425143452/e-998-msle-0.000214.h5')
Xscaler_right_filename = "Xscaler_right.save"
Yscaler_right_filename = "Yscaler_right.save"
Xscaler_left_filename = "Xscaler_left.save"
Yscaler_left_filename = "Yscaler_left.save"
Xscaler_neck_filename = "Xscaler_neck.save"
Yscaler_neck_filename = "Yscaler_neck.save"

Xscaler_right = joblib.load(Xscaler_right_filename)
Yscaler_right = joblib.load(Yscaler_right_filename)
Xscaler_left  = joblib.load(Xscaler_left_filename)
Yscaler_left  = joblib.load(Yscaler_left_filename)
Xscaler_neck  = joblib.load(Xscaler_neck_filename)
Yscaler_neck  = joblib.load(Yscaler_neck_filename)

dataset_type = 'test'
root_folder = '/media/osama/My-book/IsoGD/IsoGD_phase_1/'
gesture_root = root_folder+dataset_type+'_Gesture_descriptors/'
json_folder = root_folder+dataset_type+'_JSON_files/*'
isogd_train_video_folder_name = root_folder+dataset_type+'/'
isogd_video_folder_name = isogd_train_video_folder_name + "*"

train_folder_paths = glob(isogd_video_folder_name)
train_folder_paths.sort()

sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model_right.compile(loss='mean_squared_error', optimizer=sgd, metrics=['msle'])
model_left.compile(loss='mean_squared_error', optimizer=sgd, metrics=['msle'])
model_neck.compile(loss='mean_squared_error', optimizer=sgd, metrics=['msle'])

out_img_rows = 1080
out_img_cols = 1440
adjust_height = 200
fix_distnce_to_rescale_pts = 1500

for i in range(0, np.size(train_folder_paths)):
    M_find = []
    K_find = []
    train_videos = train_folder_paths[i] + '/*' # 000 to 180 folders in this case
    gesture_subfolder = gesture_root + train_folder_paths[i].replace(isogd_train_video_folder_name, '')
    os.makedirs(gesture_subfolder, exist_ok=True )
    print(train_videos)
    train_video_paths = glob(train_videos)
    train_video_paths.sort()

    for j in range(0, np.size(train_video_paths)):
        M_find_temp = train_video_paths[j].replace(isogd_train_video_folder_name, '').find('M')
        if M_find_temp != -1:
            M_find.append(j)
        K_find_temp = train_video_paths[j].replace(isogd_train_video_folder_name, '').find('K')
        if K_find_temp != -1:
            K_find.append(j)
    depth_video_paths = [train_video_paths[p] for p in K_find]
    print("number of depth video paths: ", np.size(depth_video_paths))
    rgb_video_paths = [train_video_paths[p] for p in M_find]


    json_root_folder_paths = glob(json_folder)
    json_root_folder_paths.sort()    # glob gives a random order so we sort the list of folders here
    json_root = json_root_folder_paths[i] + '/*'    # 000 to 180 folders in this case
    print("json_root", json_root)

    json_sub_dirs_paths = glob(json_root)
    json_sub_dirs_paths.sort()

    # video_to_display_path = depth_video_paths.copy()
    video_to_display_path = rgb_video_paths.copy()

    for q in range(0, np.size(video_to_display_path)):
        json_sub_dirs = json_sub_dirs_paths[q] + '/*'    # ***/M_***** folders
        print ("json_sub_dirs: ", json_sub_dirs)
        json_gesture_files = glob(json_sub_dirs)
        json_gesture_files.sort()                       # Individual Json files for each video
        # # <<< Get Individual Json files for each video in each folder/subfolders
        print("Processing: ", video_to_display_path[q])
        gesture_filename = gesture_root + video_to_display_path[q].replace(isogd_train_video_folder_name, '')
        gesture_filename = gesture_filename.replace('.avi', '')
        gesture_filename = gesture_filename + '.h5'
        print("gesture_filename: ", gesture_filename)

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

        orig_skeleton_images = []
        centered_skeleton_images = []
        concatenate_image_vector = []

        cropped_right = np.zeros((int(hand_height), int(hand_width), 3), dtype = 'uint8')
        cropped_left = np.zeros((int(hand_height), int(hand_width), 3), dtype = 'uint8')

        cropped_right_vector = []
        cropped_left_vector = []

        image_vector = []
        key_pts_vector = []
        key_pts_vector_orig = []
        lh_keypts_vector = []
        rh_keypts_vector = []
        fill_list = True    # To fill image_vector for the first time only
        display_frame =  int((record_num_of_frames - 1) / 2)

        vidcap = cv2.VideoCapture(video_to_display_path[q])
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("FPS is: ", fps)
        success,orig_image = vidcap.read()
        count = 0

        # cv2.namedWindow("Main video output", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Main video output', 900,540)

        while success:
            image = cv2.resize(orig_image,(out_img_cols,out_img_rows),interpolation=cv2.INTER_LINEAR)

            print(json_gesture_files[count])
            filename = json_gesture_files[count]
            get_keypts(filename)    # we get resized_keypts as a global value from this function
            fill_or_shift(image)
            repeat_missing_pts(resized_keypts, previous_pose_val_check, key_pts_vector, joint_names=pose_joint_names)
            repeat_missing_pts(lh_keypts, previous_lh_val_check, lh_keypts_vector)
            repeat_missing_pts(rh_keypts, previous_rh_val_check, rh_keypts_vector)

            if success:
                count += 1

            if success and count >= int(record_num_of_frames - 1):   # -1 as count starts from 0
                mid_array_keypoints = apply_gaussian(key_pts_vector)
                mid_array_lh_keypoints = apply_gaussian(lh_keypts_vector)
                mid_array_rh_keypoints = apply_gaussian(rh_keypts_vector)

                non_zero_mid_array = mid_array_keypoints[np.all(mid_array_keypoints != 0, axis=1)]

                output_img = np.zeros((int(out_img_rows), int(out_img_cols), 3), dtype = 'uint8')

                display_image = image_vector[display_frame].copy()
                image_to_crop = image_vector[display_frame].copy()

                draw_skeleton(display_image, mid_array_keypoints)

                top_right_joints_selected = [0, 1, 2, 3, 4, 5]  # openpose joint index
                top_left_joints_selected  = [0, 1, 2, 5, 6, 7]
                top_joints_selected  = [0, 1, 2, 3, 4, 5, 6, 7]

                upper_right_pts, missing_upper_right = get_selected_joints(top_right_joints_selected, mid_array_keypoints)
                upper_left_pts,  missing_upper_left  = get_selected_joints(top_left_joints_selected, mid_array_keypoints)
                upper_pts,  missing_upper  = get_selected_joints(top_joints_selected, mid_array_keypoints)

                # then apply fitline() function
                average_grid_size = 5   # try to keep it odd

                if np.count_nonzero(upper_right_pts) > 4:   # > 4 atleast 2 points for fit_line
                    r_vx, r_vy, r_x, r_y = fit_line(upper_right_pts, display_image=display_image)
                    right_shoulder_mean_depth  = get_mean_depth(2, mid_array_keypoints, average_grid_size)
                    right_elbow_mean_depth  = get_mean_depth(3, mid_array_keypoints, average_grid_size)
                    right_wrist_mean_depth = get_mean_depth(4, mid_array_keypoints, average_grid_size)
                    right_vec, right_link_distances = get_link_distances(top_right_joints_selected, missing_upper_right, mid_array_keypoints)
                    right_vec_index = [0, 5, 9, 12, 8 ] # get it by printing combination_idx # vector index # P2,P1 # P2 - P1
                    right_side_angles = get_angles(right_vec_index, right_vec, r_vx, r_vy, 0)
                    right_final_descriptors = get_feature_vector(right_vec, right_link_distances, right_side_angles)

                    X_right = Xscaler_right.transform(right_final_descriptors.reshape(1, 54))
                    y_right = model_right.predict(X_right.reshape(1,54,1))
                    y_right = Yscaler_right.inverse_transform(y_right)

                    right_hand_pos_mean = get_boudingbox(mid_array_keypoints, mid_array_rh_keypoints, right_vec, right_link_distances, 12)

                    create_feature_vector(right_final_descriptors, right_wrist_mean_depth, \
                    right_vec, mid_array_keypoints, display_image, image_to_crop, 12,)

                    cv2.namedWindow("cropped right", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("cropped right", int(hand_height), int(hand_width))
                    cv2.imshow("cropped right", cropped_right)  # cropped_right is a global variable
                else:
                    cropped_right_vector.append(np.zeros((int(hand_height), int(hand_width), 3), dtype = 'uint8'))

                if np.count_nonzero(upper_left_pts) > 4:
                    l_vx, l_vy, l_x, l_y = fit_line(upper_left_pts, display_image=display_image)
                    left_shoulder_mean_depth  = get_mean_depth(5, mid_array_keypoints, average_grid_size)
                    left_elbow_mean_depth  = get_mean_depth(6, mid_array_keypoints, average_grid_size)
                    left_wrist_mean_depth  = get_mean_depth(7, mid_array_keypoints, average_grid_size)
                    left_vec,  left_link_distances  = get_link_distances(top_left_joints_selected,  missing_upper_left, mid_array_keypoints)
                    left_vec_index =  [0, 5, 6, 12, 14]
                    left_side_angles  = get_angles(left_vec_index,  left_vec,  l_vx, l_vy, 1)
                    left_final_descriptors  = get_feature_vector(left_vec, left_link_distances, left_side_angles)

                    X_left =  Xscaler_left.transform(left_final_descriptors.reshape (1, 54))
                    y_left = model_left.predict(X_left.reshape(1,54,1))
                    y_left = Yscaler_left.inverse_transform(y_left)

                    left_hand_pos_mean = get_boudingbox(mid_array_keypoints, mid_array_lh_keypoints, left_vec, left_link_distances, 14)

                    create_feature_vector(left_final_descriptors, left_wrist_mean_depth, \
                    left_vec, mid_array_keypoints, display_image, image_to_crop, 14,)

                    cv2.namedWindow("cropped left", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("cropped left", int(hand_height), int(hand_width))
                    cv2.imshow("cropped left", cropped_left)  # cropped_left is a global variable
                else:
                    cropped_left_vector.append(np.zeros((int(hand_height), int(hand_width), 3), dtype = 'uint8'))

                if np.count_nonzero(upper_pts) > 4:
                    u_vx, u_vy, u_x, u_y = fit_line(upper_pts, display_image=display_image, show=False)
                    neck_mean_depth  = get_mean_depth(1, mid_array_keypoints, average_grid_size)
                    upper_vec,  upper_link_distances  = get_link_distances(top_joints_selected,  missing_upper, mid_array_keypoints)
                    upper_vec_index = [0, 7, 13, 18, 10, 25, 27]
                    upper_angles      = get_angles(upper_vec_index, upper_vec, u_vx, u_vy, 2)
                    upper_final_descriptors = get_feature_vector(upper_vec, upper_link_distances, upper_angles)

                    X_upper =  Xscaler_neck.transform(upper_final_descriptors.reshape (1, 97))
                    y_upper = model_neck.predict(X_upper.reshape(1,97,1))
                    y_upper = Yscaler_neck.inverse_transform(y_upper)

                    mask = mid_array_keypoints!=0
                    mask = mask.astype(np.int)
                    sub_mask = mask * mid_array_keypoints[1,:] # neck keypoint
                    centered_keypts = mid_array_keypoints - sub_mask
                    add_mask = mask * [(out_img_cols / 2), (out_img_rows / 2) - adjust_height, 0]
                    rescaled_keypts = centered_keypts * (y_upper / fix_distnce_to_rescale_pts)
                    display_keypts = rescaled_keypts + add_mask

                    orig_skeleton_img = show_extra_skeleton(mid_array_keypoints)
                    orig_skeleton_img = cv2.resize(orig_skeleton_img,(320, 240), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("resized image of skeleton", orig_skeleton_img)
                    # print("Shape of output_img: ", np.shape(orig_skeleton_img))
                    if count == 1:
                        orig_skeleton_images = orig_skeleton_img
                    else:
                        orig_skeleton_images.append(orig_skeleton_img)
                    # print('Size of Skeleton Images: ', np.shape(orig_skeleton_images))

                    mask_vector = np.repeat(mask[:, :, np.newaxis], 7, axis=2)
                    mask_vector = np.moveaxis(mask_vector, -1, 0)

                    # http://www.pythoninformer.com/python-libraries/numpy/index-and-slice/ Case 2
                    sub_mask_vector = np.reshape(key_pts_vector[:, 1], (7,1,3))
                    sub_mask_vector = np.repeat(sub_mask_vector, 25, axis = 1)
                    sub_mask_vector = sub_mask_vector * mask_vector

                    all_centered = key_pts_vector - sub_mask_vector
                    all_rescaled_keypts = all_centered * (y_upper / fix_distnce_to_rescale_pts)

                    add_mask_vector = mask_vector * [(out_img_cols / 2), (out_img_rows / 2) - adjust_height, 0]
                    display_keypts_vector = all_rescaled_keypts + add_mask_vector
                    # print("display frame: ", display_frame)

                    velocities = (all_rescaled_keypts[display_frame+1, :, :2] - all_rescaled_keypts[display_frame-1, :, :2]) * fps
                    accelerations = (all_rescaled_keypts[display_frame+2, :, :2] + all_rescaled_keypts[display_frame-2, :, :2] - 2*all_rescaled_keypts[display_frame, :, :2]) * fps

                    rescaled_upper_pts,  _  = get_selected_joints(top_joints_selected, rescaled_keypts) # We don't update missing upper here.
                    rescaled_upper_vec,  rescaled_upper_link_distances  = get_link_distances(top_joints_selected,  missing_upper, rescaled_keypts)
                    rescaled_upper_final_descriptors = get_feature_vector(rescaled_upper_vec, rescaled_upper_link_distances, upper_angles, velocities=velocities[:len(top_joints_selected), :], accelerations=accelerations[:len(top_joints_selected), :])

                    # for text_idx in range(0, 8):
                    #     cv2.putText(output_img,str(velocities[text_idx]),(int(display_keypts[text_idx,0]), int(display_keypts[text_idx,1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 3, cv2.LINE_AA)

                    # Calculate velocities and acceleration of upperbody joints
                    # create centered keypoints for all frames
                    # rescale of them
                    # then first derivated for the velocity
                    # and second derivate for the acceleration

                    draw_skeleton(output_img, display_keypts)
                    centered_skeleton_img = cv2.resize(output_img,(320, 240), interpolation=cv2.INTER_LINEAR)
                    if count == 1:
                        centered_skeleton_images = centered_skeleton_img
                    else:
                        centered_skeleton_images.append(centered_skeleton_img)
                    # print('Size of Centered Skeleton Images: ', np.shape(centered_skeleton_images))

                    cv2.namedWindow("Rescaled Keypoints", cv2.WINDOW_NORMAL)
                    cv2.imshow("Rescaled Keypoints", output_img)

                    # print(rescaled_upper_final_descriptors)

                    create_feature_vector(rescaled_upper_final_descriptors, neck_mean_depth, \
                                            rescaled_upper_vec, rescaled_keypts, display_image, image_to_crop, 0)
                else:
                    fill_empty = np.zeros((129,1))
                    if len(neck_feature_vector) == 0:
                        neck_feature_vector = fill_empty
                    else:
                        neck_feature_vector = np.concatenate((neck_feature_vector, fill_empty), axis = 1)

                # # >>> To show all images
                # print("Shape of image_vector: ", np.shape(image_vector))
                # for ii in range(0, record_num_of_frames):
                #     window_title = "Video Output" + str(ii)
                #     cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                #     cv2.resizeWindow(window_title, 450,270)
                #     if ii != display_frame:
                #         cv2.imshow(window_title, image_vector[ii])
                #     else:
                #         cv2.imshow(window_title, display_image)
                # # <<< To show all images
                # cv2.imshow('Main video output', display_image)
                output_image = concatenate_images(image_vector, display_image, key_pts_vector_orig)
                if count == 1:
                    concatenate_image_vector = output_image
                else:
                    concatenate_image_vector.append(output_image)
                cv2.namedWindow('Concatenated All', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Concatenated All', 1728,234)
                cv2.imshow('Concatenated All', output_image)

                k = cv2.waitKey(1)
                if k==27:    # Esc key to stop
                    cv2.destroyAllWindows()
                    print("Escape pressed... Program Closing!")
                    exit()
            success,orig_image = vidcap.read()
        # >>> Saves the model weights after each epoch if the validation loss decreased
        print("One Video Processing ends here!")
        print("Size of cropped_left_vector: ", np.shape(cropped_left_vector))
        print("Size of cropped_right_vector: ", np.shape(cropped_right_vector))
        print("Size of Upper Feature Vector: ", np.shape(neck_feature_vector))

        print("Writing data into h5 file")
        file_h5 = h5py.File(gesture_filename, 'w')
        file_h5.create_dataset('cropped_left_vector', data = cropped_left_vector)
        file_h5.create_dataset('cropped_right_vector', data = cropped_right_vector)
        file_h5.create_dataset('neck_feature_vector', data = neck_feature_vector)
        file_h5.create_dataset('centered_skeleton_images', data = centered_skeleton_images)
        file_h5.create_dataset('orig_skeleton_images', data = orig_skeleton_images)
        file_h5.create_dataset('concatenate_image_vector', data = concatenate_image_vector)
        file_h5.create_dataset('fps', data = fps)
        file_h5.close()
        print("Data written")
        #
        #
        # For reading h5py file
        print("Reading data into h5 file")
        read_h5 = h5py.File(gesture_filename, 'r')
        read_cropped_left_vector = read_h5['cropped_left_vector'][:]
        read_cropped_right_vector = read_h5['cropped_right_vector'][:]
        read_neck_feature_vector = read_h5['neck_feature_vector'][:]
        centered_skeleton_images = read_h5['centered_skeleton_images'][:]
        orig_skeleton_images = read_h5['orig_skeleton_images'][:]
        concatenate_image_vector = read_h5['concatenate_image_vector'][:]
        fps = read_h5['fps'][()]
        read_h5.close()
        print("Data read")
        # np.allclose(total_images, read_train_images)
        print('Read Feature Vector: ', np.shape(read_cropped_left_vector))
        print('Read Mean Depth Vector: ', np.shape(read_cropped_right_vector))
        print('Read Neck Feature Vector: ', np.shape(read_neck_feature_vector))
        print('Read centered_skeleton_images: ', np.shape(centered_skeleton_images))
        print('Read orig_skeleton_images: ', np.shape(orig_skeleton_images))
        print('Read concatenate_image_vector: ', np.shape(concatenate_image_vector))
        print('Read fps: ', np.shape(fps))
cv2.destroyAllWindows()
