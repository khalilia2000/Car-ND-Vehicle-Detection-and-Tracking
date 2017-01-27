# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:40:05 2017

@author: Ali.Khalili
"""

import glob
import cv2
from skimage.feature import hog
import numpy as np



# path to the working repository
work_path = 'C:/Users/ali.khalili/Desktop/Car-ND/CarND-P5-Vehicle-Detection-and-Tracking/'
# path to the final non-vehicle dataset of 64 x 64 images
non_vehicle_path = work_path + 'non-vehicles-dataset-final/'
# path to the final vehicle dataset of 64 x 64 images
vehicle_path = work_path + 'vehicles-dataset-final/'



def read_datasets():
    '''
    Read in cars and non-cars datasets
    '''
    
    file_formats = ['*.jpg', '*.png']
    # Initialize vehicle array
    vehicles = []
    for file_format in file_formats:
        file_names = glob.glob(vehicle_path+file_format)
        for file_name in file_names:
            img = cv2.imread(file_name)    
            vehicles.append(img)
    
    # Initialize non-vehicles array
    non_vehicles = []
    for file_format in file_formats:
        file_names = glob.glob(non_vehicle_path+file_format)
        for file_name in file_names:
            img = cv2.imread(file_name)    
            non_vehicles.append(img)
    
    return vehicles, non_vehicles
    


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    '''
    Return hog features and visuatlization
    orient: orientation bins (parameter to hog function)
    pix_per_call: (parameter to hog function)
    cells_per_block: (paramter to hog function)
    vis: visualize (paremter to hog function)
    feature_ved: feature_vector (parameter to hog function)
    '''
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
        
        
  
def bin_spatial(img, size=(32, 32)):
    '''
    Compute binned color features
    size: resizing the image to this size squared
    '''
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features        



# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''
    Computer color histogram features
    nbins: number of bins
    bins_range: range of values for each bin
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
    


def extract_features(imgs, color_space='BGR', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    '''
    Extract features from a list of images
    color_space: expected color space of the image
    spatial size: passed to bin_spatial()
    hist_bins: passed to color_hist()
    pix_per_cell, cell_per_block, orient: passed to get_hog_features()
    hog_channel: 0, 1, 2 or 'ALL', indicates which image channel to be passed on to get_hot_features()
    spatial_feat: if True calls bin_spatial()
    hist_feat: if True calls color_hist()
    hog_feat: if True calls get_hog_features()
    '''    
    # Create a list to append feature vectors to
    all_images_features = []
    # Iterate through the list of images
    for image in imgs:
        single_image_features = []
        # Apply color conversion if other than 'RGB'
        color_space_change_code = eval('cv2.COLOR_'+color_space+'2RGB')
        if color_space != 'RGB':
            feature_image = cv2.cvtColor(image, color_space_change_code)
        else: 
            feature_image = np.copy(image)      
        
        # Extract spatial features
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            single_image_features.append(spatial_features)
        # Extract color features in histogram form
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            single_image_features.append(hist_features)
        # Extract hog features
        if hog_feat == True:
            # Check for the image channel to be passed on to hog_features function
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            single_image_features.append(hog_features)
        
        all_images_features.append(np.concatenate(single_image_features))
    # Return list of feature vectors
    return all_images_features

    

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    Takes an image, start and stop positions in both x and y, window size in x and y directions
    and overlap fraction in both x and y directions and returns list of window coordinates.
    img: subject image
    x_start_stop: start and end position in x direction (array of size 2)
    y_start_stop: start and end position in y direction (array of size 2)
    xy_window: size of the window in both x and y directions (tuple)
    xy_overlap: amount of overlap between windows in both x and y directions (tuple)
    '''
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
    


def search_windows(img, windows, clf, scaler, color_space='BGR', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    '''
    Search image using windows and assess the predictions
    color_space: expected color space of the image
    spatial size: passed to bin_spatial()
    hist_bins: passed to color_hist()
    pix_per_cell, cell_per_block, orient: passed to get_hog_features()
    hog_channel: 0, 1, 2 or 'ALL', indicates which image channel to be passed on to get_hot_features()
    spatial_feat: if True calls bin_spatial()
    hist_feat: if True calls color_hist()
    hog_feat: if True calls get_hog_features()
    '''
    #Create an empty list to receive positive detection windows
    on_windows = []
    #Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        # Extract features for that window using single_img_features()
        features = extract_features([test_img], color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features[0]).reshape(1, -1))
        # Predict using your classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # Return windows for positive detections
    return on_windows
    
    
 
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    Draw bounding boxes on the img and return the resulting image
    '''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    

def main():
    pass



if __name__ =='__main__':
    main()