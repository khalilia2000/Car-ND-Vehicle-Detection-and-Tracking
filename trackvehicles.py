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
from helperfunctions import visualize_search_windows_on_test_images

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip
import glob
import os



# Define global variables
# frame/image objects
num_frames_to_keep = 6  # number of frames to store
recent_hot_windows = [] # list of hot windows identified on recent frames
# classifier and training related objects
svc = None              # linear SVC object
X_scaler = None         # scaler object for normalizing inputs
# hyper parameters for feature extraction
color_space = 'BGR'     # color space of the images
orient = 6              # HOG orientations
pix_per_cell = 8        # HOG pixels per cell
cell_per_block = 2      # HOG cells per block
hog_channel = 1         # Can be 0, 1, 2, or 'ALL'
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32          # Number of histogram bins
spatial_feat = True     # Spatial features on or off
hist_feat = True        # Histogram features on or off
hog_feat = True         # HOG features on or off   
# Search area coordinates and window sizes for far, mid-range and near cars
far_search_window = (np.array([[0.0,1.0], [0.5, 1.0]]), 64)
mid_search_window = (np.array([[0.0,1.0], [0.5, 1.0]]), 80)
near_search_window = (np.array([[0.0,1.0], [0.5, 1.0]]), 128)
# parameter ranges for grid_search
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, 
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


def train_classifier(vehicles_trn, 
                     vehicles_tst, 
                     non_vehicles_trn,
                     non_vehicles_tst, 
                     verbose=False, grid_search=False):
    '''
    Load images from both vehicles and non-vehicles datasets, 
    Extract features from all images,
    Normalize all features,
    Split data into train and validation sets
    Train classifier, 
    Return classifier, and scaler objects
    if verbose is True, print some details during the operations
    vehicles_trn, vehicles_tst, non_vehicles_trn, non_vehicles_tst are array of 
    images pertainnig to each set.
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
        print('Extracting features and stacking them together...')
    # Extract all features
    t0=time.time()
    vehicle_features_trn = extract_features(vehicles_trn, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    
    
    t1=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t1-t0, 2), 'Seconds to extract features from vehicle_features_trn...')
    non_vehicle_features_trn = extract_features(non_vehicles_trn, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)    
    

    t2=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t2-t1, 2), 'Seconds to extract features from non_vehicle_features_trn...')
    vehicle_features_tst = extract_features(vehicles_tst, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    
    t3=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t3-t2, 2), 'Seconds to extract features from vehicle_features_tst...')
    non_vehicle_features_tst = extract_features(non_vehicles_tst, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)    
    
    t4=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t4-t3, 2), 'Seconds to extract features from vehicle_features_tst...')
    # Stack both datasets
    X_trn = np.vstack((vehicle_features_trn, non_vehicle_features_trn)).astype(np.float64)                        
    X_tst = np.vstack((vehicle_features_tst, non_vehicle_features_tst)).astype(np.float64)                        
    
    # if verbose, print some details
    if verbose:
        print('Scaling features, creating labels, and splitting data into train and test datasets...')
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
        print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(scaled_X_trn[0]))

    t0=time.time()
    # Use a linear SVC 
    if grid_search:
        svc = GridSearchCV(SVC(C=1), tuned_parameters, cv=2, verbose=10)
    else:    
        svc = LinearSVC()    
    svc.fit(scaled_X_trn, y_trn)

    t1 = time.time()
    
    # if verbose, print some details
    if verbose:
        print(round(t1-t0, 2), 'Seconds to train SVC...')
        if grid_search:
            print(svc.best_params_)
        else:
            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(svc.score(scaled_X_tst, y_tst), 4))

    
    t_finish = time.time()
    
    # if verbose, print some details    
    if verbose:
        print('Total time: ', round(t_finish-t_start, 2), 'Seconds')
    
    


def draw_labeled_bboxes(img, labels, color=(0,0,255), thick=2):
    '''
    Draw bounding boxes around the cars identified in labels heatmap
    img: original image
    labels: result of the label fundction
    '''
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img



def mark_vehicles_on_frame(frame_img, threshold=3, verbose=False):
    '''
    Identify the vehicles in a frame and return the revised frame with vehicles identified
    with bounding boxes
    '''
    
    # Define global variables
    global recent_hot_windows
    global num_frames_to_keep
    global y_start_stop
    global far_search_window
    global mid_search_window
    global near_search_window
    
    # Identify windows that are classified as cars for all images in the recent_hot_windows
    hot_windows = []
    # Iterate through search windows that are defined globally
    for search_window in [far_search_window, mid_search_window, near_search_window]:
        # Identiry window coordinates using slide_window
        x_start_stop = ((search_window[0][0]*frame_img.shape[1]).round()).astype(int)
        y_start_stop = ((search_window[0][1]*frame_img.shape[0]).round()).astype(int)
        xy_window = (search_window[1], search_window[1])
        slide_windows = slide_window(frame_img, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                            xy_window=xy_window, xy_overlap=(0.5, 0.5))
        # Identify windows that are classified as cars                    
        hot_windows += search_windows(frame_img, slide_windows, svc, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
    # Append the results to the global list
    recent_hot_windows.append(hot_windows)
    if len(recent_hot_windows) > num_frames_to_keep:
        recent_hot_windows.pop(0)
    # Create heatmap from the hot_windows
    heatmap = np.zeros_like(frame_img)
    for frame_hot_windows in recent_hot_windows:
        for window in frame_hot_windows:
            heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    # if verbose, plot the heatmap    
    if verbose:
        plt.imshow(heatmap)
    
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    
    # Find bounding boxes around cars
    labels = label(heatmap)
    # if verbose, print some details    
    if verbose:
        print(labels[1], ' cars found')
        plt.imshow(labels[0], cmap='gray')


    # Draw the bounding boxes on the images
    draw_image = np.copy(frame_img)
    window_img = draw_labeled_bboxes(draw_image, labels, color=(0, 0, 255), thick=2)  
    
    return window_img


# path to the working repository
work_path = 'C:/Users/ali.khalili/Desktop/Car-ND/CarND-P5-Vehicle-Detection-and-Tracking/'
# path to the final non-vehicle dataset of 64 x 64 images
non_vehicle_path = work_path + 'non-vehicles-dataset-final/'
# path to the final vehicle dataset of 64 x 64 images
vehicle_path = work_path + 'vehicles-dataset-final/'
# test images
test_img_path = work_path + 'test_images/'



def process_movie(file_name):
    '''
    Load movie and replace frames with processed images and then save movie back to file
    '''
    movie_clip = VideoFileClip(work_path+file_name)
    processed_clip = movie_clip.fl_image(mark_vehicles_on_frame)
    processed_clip.write_videofile(work_path+'AK_'+file_name, audio=False, verbose=True, threads=6)



def process_test_images(sequence=False, verbose=False):
    '''
    Read test images, process them, mark the vehicles on them and save them back to the folder
    '''
    global recent_hot_windows
    # Read test images and show search rectanbles on them
    file_formats = ['*.jpg', '*.png']
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
            img_rev = mark_vehicles_on_frame(img, threshold=0, verbose=False)
            # Recorde time and print details if verbose = True
            if verbose:
                t_finish = time.time()
                print('Total time for ', os.path.basename(file_name_from), ': ', round(t_finish-t_start, 2), 'Seconds')
            # save image
            # Save image to file
            file_name_to = 'processed_'+os.path.basename(file_name_from)
            cv2.imwrite(test_img_path+file_name_to, img_rev) 





def main():
    print('reading datasets')
    a, b, c, d = read_datasets()
    print('setting up the figure')
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    print('selecting offset')
    #offset = round(np.random.uniform(len(a)-20))
    offset = 15000
    print('offset = ', offset)
    print('showing images')
    for i, ax in enumerate(axes.flat):
        ax.imshow(a[i+offset])
        ax.axis('off')



if __name__ =='__main__':
    main()
