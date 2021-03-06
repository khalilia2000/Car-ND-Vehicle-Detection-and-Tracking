# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:35:26 2017

@author: Ali.Khalili
"""

from helperfunctions import read_datasets
from helperfunctions import extract_features
from helperfunctions import slide_window
from helperfunctions import search_windows
from helperfunctions import get_hog_features

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import time
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip
import glob
import os
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from matplotlib import gridspec


# Define global variables
# frame/image objects
num_frames_to_keep = 10     # number of frames to store
recent_hot_windows = []     # list of hot windows identified on recent frames
recent_bbox_windows = []    # list of bounding boxes around cars
# classifier and training related objects
clf = None                  # classifier object
X_scaler = None             # scaler object for normalizing inputs
# hyper parameters for feature extraction
color_space = 'BGR'         # color space of the images
orient = 14                 # HOG orientations
pix_per_cell = 16           # HOG pixels per cell
cell_per_block = 3          # HOG cells per block
target_color_space='YCrCb'  # target color space
hog_channel = [0,1]         # Channels to extract hog from
spatial_size = (8, 8)       # Spatial binning dimensions
hist_bins = 16              # Number of histogram bins
spatial_feat = True         # Spatial features on or off
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off   
# Thresholds for procesing heatmaps
thresh_high=10
thresh_low=5
# Search widnows below indicates the areas of interest that should be searched plus the search window size.  
# The first element is the ((x_min, x_max), (y_min, y_max)) where the coordiantes are relative to the image
# size, (i.e. between 0 and 1) and the second element is the size of the search widnow:
search_window_0 = (np.array([[0.0,1.0], [0.5, 1.0]]), 32)
search_window_1 = (np.array([[0.0,1.0], [0.5, 1.0]]), 64)
search_window_2 = (np.array([[0.0,1.0], [0.5, 1.0]]), 96)
search_window_3 = (np.array([[0.0,1.0], [0.5, 1.0]]), 128)
all_search_windows = [search_window_1,
                      search_window_2, 
                      search_window_3]
# To keep track of the frame number during video processing for debugging purposes
frame_no=0                  # for debugging purposes
max_heat_list = []          # for debugging purposes
# path to the working repository
home_computer = False
if home_computer == True:
    work_path = 'C:/Udacity Courses/Car-ND-Udacity/P5-Vehicle-Tracking/'
else:
    work_path = 'C:/Users/ali.khalili/Desktop/Car-ND/CarND-P5-Vehicle-Detection-and-Tracking/'
# path to the final non-vehicle dataset of 64 x 64 images
non_vehicle_path = work_path + 'non-vehicles-dataset-final/'
# path to the final vehicle dataset of 64 x 64 images
vehicle_path = work_path + 'vehicles-dataset-final/'
# test images
test_img_path = work_path + 'test_images/'



def extract_features_from_datasets(vehicles_trn, vehicles_tst, non_vehicles_trn, non_vehicles_tst, verbose=False):
    '''
    Extract features from training and test datasets pertaining to both vehicles and non-vehicles
    vehicles_trn, vehicles_tst, non_vehicles_trn and non_vehicles_tst are all datasets of images
    verbose: if True, print additional details during the operation
    '''  
    # track time    
    t_start = time.time()  
    
    # if verbose, print some details
    if verbose:
        print('Extracting features and stacking them together...')
    # track time    
    t0=time.time()
    
    # Extract all features for training dataset vehicles   
    vehicle_features_trn = extract_features(vehicles_trn, 
                        source_color_space=color_space, target_color_space=target_color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, 
                        hog_feat=hog_feat)
    # track time    
    t1=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t1-t0, 2), 'Seconds to extract features from vehicle_features_trn...')
    
    # Extract all features for test dataset vehicles
    vehicle_features_tst = extract_features(vehicles_tst, 
                        source_color_space=color_space, target_color_space=target_color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat,
                        hog_feat=hog_feat)
    # track time    
    t2=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t2-t1, 2), 'Seconds to extract features from vehicle_features_tst...')
        
    # Extract all features for training dataset non-vehicles    
    non_vehicle_features_trn = extract_features(non_vehicles_trn, 
                        source_color_space=color_space, target_color_space=target_color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat,
                        hog_feat=hog_feat)    
    # track time    
    t3=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t3-t2, 2), 'Seconds to extract features from non_vehicle_features_trn...')
        
    # Extract all features for test dataset non-vehicles
    non_vehicle_features_tst = extract_features(non_vehicles_tst, 
                        source_color_space=color_space, target_color_space=target_color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat,
                        hog_feat=hog_feat)    
    # track time    
    t4=time.time()
    t_finish = time.time()
    # if verbose, print some details
    if verbose:
        print(round(t4-t3, 2), 'Seconds to extract features from non_vehicle_features_tst...')
        print(round(t_finish-t_start, 2), 'Seconds to extract all features from all datasets...')
    
    # return extracted features for each dataset
    return vehicle_features_trn, vehicle_features_tst, non_vehicle_features_trn, non_vehicle_features_tst
        
        


def train_classifier(vehicle_features_trn, 
                     vehicle_features_tst, 
                     non_vehicle_features_trn, 
                     non_vehicle_features_tst, 
                     verbose=False,
                     **clf_kwargs):
    '''
    Normalize features from all feature datasets that are passed to the function,
    Train classifier, 
    Return classifier, and scaler objects
    if verbose is True, print some details during the operations
    vehicle_features_trn, vehicle_features_tst, non_vehicle_features_trn, non_vehicle_features_tst are array of 
    features.
    '''
    # Define global variables to use in this function
    global clf    
    global X_scaler
    clf = None
    X_scaler = None
    
    # Stack vehicles and non-vehicles datasets
    X_trn = np.vstack((vehicle_features_trn, non_vehicle_features_trn)).astype(np.float64)                        
    X_tst = np.vstack((vehicle_features_tst, non_vehicle_features_tst)).astype(np.float64)                        
    # if verbose, print some details
    if verbose:
        print('Scaling features, and creating labels...')
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_trn)
    # Apply the scaler to X_trn and X_tst
    scaled_X_trn = X_scaler.transform(X_trn)
    scaled_X_tst = X_scaler.transform(X_tst)
    
    # Define the labels vector
    y_trn = np.hstack((np.ones(len(vehicle_features_trn)), np.zeros(len(non_vehicle_features_trn))))
    y_tst = np.hstack((np.ones(len(vehicle_features_tst)), np.zeros(len(non_vehicle_features_tst))))
    
    # if verbose, print some details
    if verbose:
        print('Using:',orient,'orientations, ',
              pix_per_cell,'pixels per cell, ', 
              cell_per_block,'cells per block, and hog_channel = ', hog_channel)
        print('Feature vector length:', len(scaled_X_trn[0]))
    # track time
    t0=time.time()
    # Use a Random Forest Classifier
    #clf = RandomForestClassifier(n_estimators=100, max_features=35, 
    #                             min_samples_split=100, verbose=0)
    # use a linear SVC classifier    
    clf = LinearSVC(**clf_kwargs)    
    clf.fit(scaled_X_trn, y_trn)
    # track time
    t1 = time.time()
    
    # if verbose, print some details
    if verbose:
        print(round(t1-t0, 2), 'Seconds to train classifier...')
        # Check the score of the classifier
        print('Test Accuracy of the classifier = ', round(clf.score(scaled_X_tst, y_tst), 4))

    
    

def draw_bboxes_using_watershed(img, heatmap_high, heatmap_low, 
                                color=(0,0,255), thick=2, verbose=False):
    '''
    This algorithm uses watershed algorithm to put bounding boxes around cars that are close together separately
    img: original image
    heatmap_high: the heatmap that is created from on_windows using a high threshold
    heatmap_low: the heatmap that is created from on_windows using a low threshold
    color: color of the bounding box 
    thick: thickness of the bounding bix
    verbose: verbosity of the operation
    '''
    # Create masked image and calculate distance for the high thresholded heatmap using single channel
    masked_img_high = np.copy(heatmap_high[:,:,0])
    masked_img_high[masked_img_high>0]=1
    distance_high = ndi.distance_transform_edt(masked_img_high)  # Calculate distance from the background
    
    # Create masked image and calculate distance for the low thresholded heatmap using single channel
    masked_img_low = np.copy(heatmap_low[:,:,0])
    masked_img_low[masked_img_low>0]=1
    distance_low = ndi.distance_transform_edt(masked_img_low)  # Calculate distance from the background    
    
    # Calculate local maxima and convert into dtype=int
    local_maxima = peak_local_max(distance_high, indices=False, footprint=np.ones((144,144)))
    local_maxima = local_maxima.astype(int)
    # Use label function to identiry and label various local maxima that is found
    markers = label(local_maxima)[0]
    
    # Use watershed algorithm to identify various portions of the image and assume each one is a car
    labels = watershed(-distance_low, markers, mask=masked_img_low)
    
    # Identify the number of cars found
    n_cars = labels.max()
    # if verbose, print some details    
    if verbose:
        print(n_cars, ' cars found')
        cv2.imwrite(test_img_path+'pipeline_5.jpg', distance_high*255.0/distance_high.max())
        cv2.imwrite(test_img_path+'pipeline_6.jpg', distance_low*255.0/distance_low.max())        
        cv2.imwrite(test_img_path+'pipeline_7.jpg', labels*100)
    # keep a list of all bounding boxes that are drawn
    bbox_list=[]
    # Iterate through all detected cars
    for car_number in range(1, n_cars+1):
        # Find pixels with each car_number label value
        nonzero = (labels == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
        
    # Return the image
    return img, bbox_list



def draw_bboxes_using_label(img, heatmap, color=(0,0,255), thick=2, verbose=False):
    '''
    Draw bounding boxes around the cars identified in labels heatmap
    img: original image
    color: color of the bounding box 
    thick: thickness of the bounding bix
    verbose: verbosity of the operation
    '''
    # Create labels from the heatmap
    labels = label(heatmap)
    # if verbose, print some details    
    if verbose:
        print(labels[1], ' cars found')
        print('maximum intensity of heatmap = ', heatmap.max())
        cv2.imwrite(test_img_path+'pipeline_5.jpg', labels[0]*100)
    # keep a list of all bounding boxes that are drawn
    bbox_list=[]
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img, bbox_list



def histogram_equalize(img):
    '''
    equalize histogram for all image channels
    '''
    ch0, ch1, ch2 = cv2.split(img)
    ch0_eq = cv2.equalizeHist(ch0)
    ch1_eq = cv2.equalizeHist(ch1)
    ch2_eq = cv2.equalizeHist(ch2)
    return cv2.merge((ch0_eq, ch1_eq, ch2_eq))



def mark_vehicles_on_frame(frame_img, verbose=False, plot_heat_map=False, plot_box=True, watershed=True, 
                           batch_hog=True, debug=False):
    '''
    Identify the vehicles in a frame and return the revised frame with vehicles identified
    with bounding boxes
    frame_img: the frame image to be revised
    verbose: determine the verbosity of the operation
    plot_heat_map: plots the heatmap on the frame
    plot_box: plots bounding boxes on the frame
    watershed: uses the watershed algorithm for identifying cars otherwise only uses label
    batch_hog: uses batch_hog algorithm to speed up the process of hog feature extraction
    debug: debug mode
    '''
    
    # Define global variables
    global frame_no    
    global recent_hot_windows
    global recent_bbox_windows
    global max_heat_list
    # Identify windows that are classified as cars for all images in the recent_hot_windows
    hot_windows = []
    # Iterate through search windows that are defined globally
    for search_window in all_search_windows:
        # Identiry window coordinates using slide_window
        x_start_stop = ((search_window[0][0]*frame_img.shape[1]).round()).astype(int)
        y_start_stop = ((search_window[0][1]*frame_img.shape[0]).round()).astype(int)
        xy_window = (search_window[1], search_window[1])
        slide_windows = slide_window(frame_img.shape, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
        # Identify windows that are classified as cars                    
        hot_windows += search_windows(frame_img, search_window, slide_windows, clf, X_scaler, 
                                batch_hog=batch_hog, 
                                source_color_space=color_space, 
                                target_color_space=target_color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, 
                                hog_feat=hog_feat)
    # if verbose, save some photos    
    if verbose:
        pipeline_1 = np.copy(frame_img)
        for window in hot_windows:
            cv2.rectangle(pipeline_1, window[0], window[1], color=(0,0,255), thickness=2)
        cv2.imwrite(test_img_path+'pipeline_1.jpg', pipeline_1) 
    
    # Append the results to the global list
    recent_hot_windows.append(hot_windows)
    if len(recent_hot_windows) > num_frames_to_keep:
        recent_hot_windows.pop(0)
    # Create heatmap from the hot_windows
    heatmap = np.zeros_like(frame_img)
    for frame_hot_windows in recent_hot_windows:
        for window in frame_hot_windows:
            heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    # Normalize the pixel values
    heatmap = cv2.convertScaleAbs(heatmap, heatmap, 1/len(recent_hot_windows))    
    
    # if verbose, plot the heatmap and save to file   
    if verbose:
        scaled_heatmap = np.copy(heatmap)
        scaled_heatmap = cv2.convertScaleAbs(heatmap,scaled_heatmap,255/heatmap.max())
        scaled_heatmap[:,:,:2] = 0
        cv2.imwrite(test_img_path+'pipeline_2.jpg', cv2.addWeighted(np.copy(frame_img), 1, scaled_heatmap, 0.7, 0))
    
    # Zero out pixels below the threshold and construct both high and low thresholded heatmaps
    heatmap_high = np.copy(heatmap)
    heatmap_high[heatmap_high <= thresh_high] = 0    
    heatmap_low = np.copy(heatmap)
    heatmap_low[heatmap_low <= thresh_low] = 0     
    # if verbose, save some photos and print some details   
    if verbose:
        print('maximum intensity of heatmap_high = ', heatmap_high.max())
        # save heatmap_high
        scaled_heatmap_high = np.copy(heatmap_high)
        scaled_heatmap_high = cv2.convertScaleAbs(heatmap_high,scaled_heatmap_high,255/heatmap_high.max())
        scaled_heatmap_high[:,:,:2] = 0
        cv2.imwrite(test_img_path+'pipeline_3.jpg', cv2.addWeighted(np.copy(frame_img), 1, scaled_heatmap_high, 0.7, 0))
        # save heatmap_low
        scaled_heatmap_low = np.copy(heatmap_low)
        scaled_heatmap_low = cv2.convertScaleAbs(heatmap_low,scaled_heatmap_low,255/heatmap_low.max())
        scaled_heatmap_low[:,:,:2] = 0
        cv2.imwrite(test_img_path+'pipeline_4.jpg', cv2.addWeighted(np.copy(frame_img), 1, scaled_heatmap_low, 0.7, 0))
    
    # Draw the bounding boxes on the images
    draw_image = np.copy(frame_img)
    if plot_box:
        draw_color = [0,0,0]
        draw_color[color_space.index('B')] = 255
        draw_color = tuple(draw_color)
        if watershed:
            draw_image, bbox_list = draw_bboxes_using_watershed(draw_image, heatmap_high, heatmap_low, 
                                                                color=draw_color, thick=1, verbose=verbose) 
        else:
            draw_image, bbox_list = draw_bboxes_using_label(draw_image, heatmap_high, color=draw_color, thick=1, verbose=verbose) 
    
    # keep track of the bounding goxes in the most recent frames
    recent_bbox_windows.append(bbox_list)
    if len(recent_bbox_windows) > num_frames_to_keep:
        recent_bbox_windows.pop(0)
    
    # plot heatmap on frame
    if plot_heat_map:
        scaled_heatmap_low = heatmap_low*100
        scaled_heatmap_low[scaled_heatmap_low>255] = 255
        scaled_heatmap_low[:,:,:2] = 0
        draw_image = cv2.addWeighted(draw_image, 1, scaled_heatmap_low, 0.5, 0)
    # Save individual frames for debugging purposes if required
    if debug:
        print(heatmap.max())
        scaled_heatmap = np.copy(heatmap)
        scaled_heatmap = cv2.convertScaleAbs(heatmap,scaled_heatmap,255/heatmap.max())
        scaled_heatmap[:,:,:2] = 0
        cv2.imwrite(work_path+'tmp/frame_{:04d}.png'.format(frame_no), 
                    cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(work_path+'tmp/heatmap_{:04d}.jpg'.format(frame_no), 
                    cv2.cvtColor(cv2.addWeighted(np.copy(draw_image), 1, scaled_heatmap, 0.7, 0),cv2.COLOR_RGB2BGR))
    frame_no+=1
    
    return draw_image



def process_movie(file_name, pre_fix='AK_', high_threshold=20, low_threshold=15, c_space='RGB'):
    '''
    Load movie and replace frames with processed images and then save movie back to file
    '''
    global thresh_high
    global thresh_low
    global color_space
    global recent_hot_windows
    thresh_high = high_threshold
    thresh_low = low_threshold
    color_space = c_space
    recent_hot_windows = []
    
    movie_clip = VideoFileClip(work_path+file_name)
    processed_clip = movie_clip.fl_image(mark_vehicles_on_frame)
    processed_clip.write_videofile(work_path+pre_fix+file_name, audio=False, verbose=True, threads=6)




def process_test_images(sequence=False, verbose=False, high_threshold=10, low_threshold=5, watershed=True, batch_hog=True):
    '''
    Read test images, process them, mark the vehicles on them and save them back to the folder
    sequence: is image part of sequence of frames? if True, the recent_hot_windows will be updated like a movie stream
    verbose: verbosity of the operation
    high_threshold: the value of the high_threshold
    low_threshold: the value of the low threshold
    watershed: if True uses the watershed algorithm for identifying cars otherwise only uses label
    batch_hog: if True uses a faster HOG extraction algorithm
    '''
    global recent_hot_windows
    global thresh_high
    global thresh_low
    global num_frames_to_keep
    thresh_high = high_threshold
    thresh_low = low_threshold
    
    # Read test images and show search rectanbles on them
    file_formats = ['*.jpg', '*.png', '*.jpeg']
    # Iterate through files
    for file_format in file_formats:
        file_names = glob.glob(test_img_path+file_format)
        for file_name_from in file_names:
            # Load image
            img = cv2.imread(file_name_from) 
            # Recorde time if verbose = True
            if verbose:
                t_start = time.time()
            # recent rect_hot_windows each time if not processing sequnce images
            if not sequence:
                recent_hot_windows = []
            # process image
            img_rev = mark_vehicles_on_frame(img, verbose=verbose, watershed=watershed, batch_hog=batch_hog)
            # Recorde time and print details if verbose = True
            if verbose:
                t_finish = time.time()
                print('Total time for ', os.path.basename(file_name_from), ': ', round(t_finish-t_start, 2), 'Seconds')
            # Save image to file
            file_name_to = 'processed_'+os.path.basename(file_name_from)
            cv2.imwrite(test_img_path+file_name_to, img_rev) 

    

def read_data_and_train_classifier(limit_trn=-1, random=False, verbose=True, **clf_kwargs):
    '''
    reset clf and X_scaler variables, read training data and train the classifier from scratch
    limit_trn: if -1 uses all the training data availble, otherwise the number of training samples to beused.
    random: use random shuffling of the train / test datasets if True
    verbose: verbosity of the operation
    **clf_kwargs: parameters to be passed on to the classifier during training
    '''
    # read data
    print('reading datasets')
    v_trn, v_tst, nv_trn, nv_tst = read_datasets(limit_trn=limit_trn, random=random)
    feat_v_trn, feat_v_tst, feat_nv_trn, feat_nv_tst = extract_features_from_datasets(v_trn[0], v_tst[0], 
                                                                                      nv_trn[0], nv_tst[0], 
                                                                                      verbose=verbose)
    # train classifier
    train_classifier(feat_v_trn, feat_v_tst, feat_nv_trn, feat_nv_tst, verbose=verbose, **clf_kwargs)
    # write fnames to file for furthe exploration
    df_trn_veh = pd.DataFrame([os.path.basename(item) for item in v_trn[1]], columns=['trn_veh'])
    df_tst_veh = pd.DataFrame([os.path.basename(item) for item in v_tst[1]], columns=['tst_veh'])
    df_trn_nvh = pd.DataFrame([os.path.basename(item) for item in nv_trn[1]], columns=['trn_nvh'])
    df_tst_nvh = pd.DataFrame([os.path.basename(item) for item in nv_tst[1]], columns=['tst_nvh'])
    df_list = [df_trn_veh, df_tst_veh, df_trn_nvh, df_tst_nvh]
    df_combined = pd.concat(df_list, ignore_index=True, axis=1)
    writer = pd.ExcelWriter('file_names.xlsx')
    df_combined.to_excel(writer,'Sheet1')
    writer.save()
    writer.close()
    
    # return feature datasets
    return feat_v_trn, feat_v_tst, feat_nv_trn, feat_nv_tst



def save_to_file(clf_fname, xscaler_fname):
    '''    
    Save clf and X_scaler to file
    clf_fname: filename for clf 
    xscaler_fname: filename for xscaler
    '''
    joblib.dump(clf, clf_fname)
    joblib.dump(X_scaler, xscaler_fname)



def load_from_file(clf_fname, xscaler_fname):
    '''    
    Save clf and X_scaler to file
    clf_fname: filename for clf 
    xscaler_fname: filename for xscaler
    '''
    global clf
    global X_scaler
    clf = joblib.load(work_path+clf_fname)
    X_scaler = joblib.load(work_path+xscaler_fname)



def plot_hog_images(data_tuple, indices=None, num_images=5, cs_list=['RGB','HSV','YCrCb','LUV','Gray']):
    '''
    plots random hog features images from the dataset that is passed on to this function.
    data_tupe: 2-tuple with first element containing the images, and 2nd element containnig the filenames.
    num_images: number of images to choose from the dataset. only used if indices is None
    indices: indices of the photos to use, randomly choose if None
    cs_list: list of color spaces
    '''
    # creating the grid space
    hspace = 0.2    # distance between images vertically
    wspace = 0.01   # distance between images horizontally
    if indices is None:
        n_rows = num_images
    else:
        n_rows = len(indices)
    n_cols = len(cs_list)*3+1
    g_fig = gridspec.GridSpec(n_rows,n_cols) 
    g_fig.update(wspace=wspace, hspace=hspace)
    
    # setting up the figure
    size_factor = 4.5
    aspect_ratio = data_tuple[0][0].shape[1]/data_tuple[0][0].shape[0]
    fig_w_size = n_cols*size_factor*aspect_ratio+(n_cols-1)*wspace
    fig_h_size = n_rows*size_factor+(n_rows-1)*hspace
    plt.figure(figsize=(fig_w_size,fig_h_size))
        
    ax_list = []
    counter = 0
    if indices is None:
        indices = np.random.choice(len(data_tuple[0]), num_images)
    for idx, ch_ind in enumerate(indices):
        img = data_tuple[0][ch_ind]
        fname = data_tuple[1][ch_ind]
        ax_list.append(plt.subplot(g_fig[counter]))
        ax_list[-1].imshow(img)
        ax_list[-1].axis('off')
        ax_list[-1].set_title(os.path.basename(fname))
        counter+=1
        for cs in cs_list:
            if cs != color_space:
                color_space_change_code = eval('cv2.COLOR_'+color_space+'2'+cs)
                rev_img = cv2.cvtColor(img, color_space_change_code)
            else:
                rev_img = np.copy(img)
            for ch in range(rev_img.shape[2]):
                feat, hog_img = get_hog_features(rev_img[:,:,ch], orient, pix_per_cell, 
                                                 cell_per_block, vis=True, feature_vec=True)
                ax_list.append(plt.subplot(g_fig[counter]))    
                ax_list[-1].imshow(hog_img, cmap='gray')
                ax_list[-1].set_title('color_space='+cs+' - CH='+str(ch))
                ax_list[-1].axis('off')
                counter+=1
    plt.savefig(test_img_path+'hog_feat_plot.png')
    
    return indices
    


def main():
    pass


if __name__ =='__main__':
    main()
