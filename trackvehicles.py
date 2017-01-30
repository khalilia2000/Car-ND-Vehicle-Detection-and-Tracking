# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:35:26 2017

@author: Ali.Khalili
"""

from helperfunctions import read_datasets
from helperfunctions import extract_features
from helperfunctions import slide_window
from helperfunctions import search_windows
from helperfunctions import draw_boxes

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time


# Define global variables
# frame/image objects
num_frames_to_keep = 6  # number of frames to store
recent_frames = []      # list of frame images that are stored
# classifier and training related objects
svc = None              # linear SVC object
X_scaler = None         # scaler object for normalizing inputs
# hyper parameters for feature extraction
color_space = 'BGR'     # color space of the images
orient = 12             # HOG orientations
pix_per_cell = 8        # HOG pixels per cell
cell_per_block = 2      # HOG cells per block
hog_channel = 'ALL'     # Can be 0, 1, 2, or 'ALL'
spatial_size = (8, 8)   # Spatial binning dimensions
hist_bins = 32          # Number of histogram bins
spatial_feat = True     # Spatial features on or off
hist_feat = True        # Histogram features on or off
hog_feat = True         # HOG features on or off   
# search window properties
y_start_stop = [450, 720] # Min and max in y to search in slide_window()


def train_classifier(verbose=False):
    '''
    Load images from both vehicles and non-vehicles datasets, 
    Extract features from all images,
    Normalize all features,
    Split data into train and validation sets
    Train classifier, 
    Return classifier, and scaler objects
    if verbose is True, print some details during the operations
    '''
    # Define global variables to use in this function
    global svc    
    global X_scaler
    global color_space
    global orient
    global pix_per_cell
    global cell_per_block
    global hog_channel
    global spatial_size
    global hist_bins
    global spatial_feat
    global hist_feat
    global hog_feat 
    
    # Track time    
    t_start = time.time()    
    
    # if verbose, print some details
    if verbose:
        print('Reading datasets...')
    # Read images from both datasets
    vehicles, non_vehicles = read_datasets()
    
    # if verbose, print some details
    if verbose:
        print('Extracting features and stacking them together...')
    # Extract all features
    vehicle_features = extract_features(vehicles, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    non_vehicle_features = extract_features(non_vehicles, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)    
    # Stack both datasets
    X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)                        
    
    # if verbose, print some details
    if verbose:
        print('Scaling features, creating labels, and splitting data into train and test datasets...')
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, 
                                                        stratify=y, random_state=rand_state)
    
    # if verbose, print some details
    if verbose:
        print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC 
    svc = LinearSVC()    
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    
    # if verbose, print some details
    if verbose:
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    
    t_finish = time.time()
    
    # if verbose, print some details    
    if verbose:
        print('Total time: ', round(t_finish-t_start, 2), 'Seconds')
    
    


def mark_vehicles_in_frame(frame_img, threshold=2):
    '''
    Identify the vehicles in a frame and return the revised frame with vehicles identified
    with bounding boxes
    '''
    # Define global variables
    global recent_frames
    global num_frames_to_keep
    global y_start_stop
    # Add frame_img to the list of recent_frames and remove the oldest one if length is exceeding the limit
    recent_frames.append(frame_img)
    if len(recent_frames) > num_frames_to_keep:
        recent_frames.pop(0)
    
    windows = slide_window(frame_img, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    
    hot_windows = search_windows(frame_img, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    
    draw_image = np.copy(frame_img)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)  
    
    return window_img


def main():
    pass



if __name__ =='__main__':
    main()