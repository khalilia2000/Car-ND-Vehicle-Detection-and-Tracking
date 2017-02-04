# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:40:05 2017

@author: Ali.Khalili
"""

import glob
import cv2
from skimage.feature import hog
import numpy as np
import os


# path to the working repository
home_computer = True
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



def base_name(x_str):
    res_str = os.path.basename(x_str)[0:os.path.basename(x_str).find('.')]
    if res_str.find('_') > 0:
        res_str = res_str[0:res_str.find('_')]
    return (len(res_str), res_str)


def read_datasets():
    '''
    Read in cars and non-cars datasets
    '''
    
    file_formats = ['*.jpg', '*.png']
    # Initialize vehicle array
    vehicles = []
    file_names_vehicles = []
    for file_format in file_formats:
        file_names_vehicles += glob.glob(vehicle_path+file_format)
    file_names_vehicles = sorted(file_names_vehicles, key=base_name)
    for file_name in file_names_vehicles:
        img = cv2.imread(file_name)    
        vehicles.append(img)
    
    # Initialize non-vehicles array
    non_vehicles = []
    file_names_non_vehicles = []
    for file_format in file_formats:
        file_names_non_vehicles += glob.glob(non_vehicle_path+file_format)
    file_names_non_vehicles = sorted(file_names_non_vehicles, key=base_name)        
    for file_name in file_names_non_vehicles:
        img = cv2.imread(file_name)    
        non_vehicles.append(img)
    
    # Randomly split the dataset into train and test datasets
    # Select test data set as a consequtive block to avoid contaminating datasets
    test_size = 0.2
    vehicle_test_size = round(test_size*len(vehicles))
    non_vehicle_test_size = round(test_size*len(non_vehicles))
    vehicle_index = round(np.random.uniform(len(vehicles)-vehicle_test_size))
    non_vehicle_index = round(np.random.uniform(len(non_vehicles)-non_vehicle_test_size))
    # Copy data into train and test datasets
    # vehicles - training set
    v_trn_imgs = vehicles[0:vehicle_index] + \
                    vehicles[vehicle_index+vehicle_test_size:]
    v_trn_fnames = file_names_vehicles[0:vehicle_index] + \
                    file_names_vehicles[vehicle_index+vehicle_test_size:]
    v_trn = (v_trn_imgs, v_trn_fnames)
    # vehicles - test set
    v_tst_imgs = vehicles[vehicle_index:vehicle_index+vehicle_test_size]
    v_tst_fnames = file_names_vehicles[vehicle_index:vehicle_index+vehicle_test_size]
    v_tst = (v_tst_imgs, v_tst_fnames)
    # non-vehicles - training set
    nv_trn_imgs = non_vehicles[0:non_vehicle_index] + \
                    non_vehicles[non_vehicle_index+non_vehicle_test_size:]
    nv_trn_fnames = file_names_non_vehicles[0:non_vehicle_index] + \
                    file_names_non_vehicles[non_vehicle_index+non_vehicle_test_size:]
    nv_trn = (nv_trn_imgs, nv_trn_fnames)
    # non-vehicles - test set
    nv_tst_imgs = non_vehicles[non_vehicle_index:non_vehicle_index+non_vehicle_test_size]
    nv_tst_fnames = file_names_non_vehicles[non_vehicle_index:non_vehicle_index+non_vehicle_test_size]
    nv_tst = (nv_tst_imgs, nv_tst_fnames)    
    
    return v_trn, v_tst, nv_trn, nv_tst
    


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
                        spatial_feat=True, hist_feat_RGB=True, hist_feat_HSV=True, hog_feat=True):    
    '''
    Extract features from a list of images
    color_space: expected color space of the image
    spatial size: passed to bin_spatial()
    hist_bins: passed to color_hist()
    pix_per_cell, cell_per_block, orient: passed to get_hog_features()
    hog_channel: 0, 1, 2 or 'ALL', indicates which image channel to be passed on to get_hot_features()
    spatial_feat: if True calls bin_spatial()
    hist_feat_RGB: if True calls color_hist() on RGB image
    hist_feat_HSV: if True calls color_hist() on HSV image
    hog_feat: if True calls get_hog_features()
    '''    
    # Create a list to append feature vectors to
    all_images_features = []
    # Iterate through the list of images
    for image in imgs:
        single_image_features = []
        # Make RGB converted image if other than 'RGB'
        color_space_change_code = eval('cv2.COLOR_'+color_space+'2RGB')
        if color_space != 'RGB':
            feature_image_RGB = cv2.cvtColor(image, color_space_change_code)
        else: 
            feature_image_RGB = np.copy(image)      
         
        # Make HSV converted image if other than 'HSV'
        color_space_change_code = eval('cv2.COLOR_'+color_space+'2HSV')
        if color_space != 'HSV':
            feature_image_HSV = cv2.cvtColor(image, color_space_change_code)
        else: 
            feature_image_HSV = np.copy(image)                  
        
        # Extract spatial features
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image_RGB, size=spatial_size)
            single_image_features.append(spatial_features)
        # Extract color features in histogram form RGB image
        if hist_feat_RGB == True:
            hist_features = color_hist(feature_image_RGB, nbins=hist_bins)
            single_image_features.append(hist_features)
        # Extract color features in histogram form from HSV image
        if hist_feat_HSV == True:
            hist_features = color_hist(feature_image_HSV, nbins=hist_bins)
            single_image_features.append(hist_features)
        # Extract hog features
        if hog_feat == True:
            # Check for the image channel to be passed on to hog_features function
            if hog_channel == 'RGB_ALL':
                hog_features = []
                for channel in range(feature_image_RGB.shape[2]):
                    hog_features.append(get_hog_features(feature_image_RGB[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            elif hog_channel == 'HSV_ALL':
                hog_features = []
                for channel in range(feature_image_HSV.shape[2]):
                    hog_features.append(get_hog_features(feature_image_HSV[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            elif hog_channel in ['R', 'G', 'B']:
                hog_features = get_hog_features(feature_image_RGB[:,:,['R', 'G', 'B'].index(hog_channel)], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            elif hog_channel in ['H', 'S', 'V']:
                hog_features = get_hog_features(feature_image_HSV[:,:,['H', 'S', 'V'].index(hog_channel)], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            single_image_features.append(hog_features)
        
        all_images_features.append(np.concatenate(single_image_features))
    # Return list of feature vectors
    return all_images_features

    

def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    Takes an image, start and stop positions in both x and y, window size in x and y directions
    and overlap fraction in both x and y directions and returns list of window coordinates.
    img_shape: subject image shape
    x_start_stop: start and end position in x direction (array of size 2)
    y_start_stop: start and end position in y direction (array of size 2)
    xy_window: size of the window in both x and y directions (tuple)
    xy_overlap: amount of overlap between windows in both x and y directions (tuple)
    '''
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shape[0]
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
    


def search_windows(img, windows_list, clf, scaler, color_space='BGR', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat_RGB=True, 
                    hist_feat_HSV=True, 
                    hog_feat=True):
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
    #Create an empty list to store all sliding windows taken from the img
    window_imgs = []
    #Iterate over all windows in the list
    for window in windows_list:
        # Extract the test window from original image
        window_imgs.append(cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)))      
    
    # Extract features for that window using single_img_features()
    features_list = extract_features(window_imgs, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat_RGB=hist_feat_RGB, 
                        hist_feat_HSV=hist_feat_HSV, 
                        hog_feat=hog_feat)
   
    # Return those windows with positive classification outcome    
    for window, features in zip(windows_list, features_list):
        # Scale extracted features to be fed to classifier
        scaled_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        prediction = clf.predict(scaled_features)
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
    
  

def visualize_search_windows_on_test_images(search_area, window_size, overlap=0.5, color=(0, 0, 255), thick=6, write_to_file=True, images=None):
    '''
    Iterate through test images and draw the search windows on the image and save to file again
    search_area : array indicating the search area [[xmin,xmax],[ymin,ymax]] in fraction of the images size
    window_size: search window size 
    overlap: overlapping fraction of search windows
    '''    
    # Read test images and show search rectanbles on them
    file_formats = ['*.jpg', '*.png']
    # Iterate through files
    imgs_rev = []
    
    if images is None:
        images = []
        for file_format in file_formats:
            file_names = glob.glob(test_img_path+file_format)
            for file_name_from in file_names:
                # Load image
                img = cv2.imread(file_name_from)    
                images.append(img)
    
    counter = 0
    for img in images:
        # Identify slide windows
        x_start_stop = ((search_area[0]*img.shape[1]).round()).astype(int)
        y_start_stop = ((search_area[1]*img.shape[0]).round()).astype(int)
        windows_list = slide_window(img.shape, 
                                    x_start_stop=x_start_stop, 
                                    y_start_stop=y_start_stop, 
                                    xy_window=(window_size, window_size), 
                                    xy_overlap=(overlap, overlap))
        # Draw boxes on the image
        img_rev = draw_boxes(img, windows_list, color=color, thick=thick)
        imgs_rev.append(img_rev)
        # Save image to file
        if write_to_file:
            counter += 1
            file_name_to = 'search_boxes_'+str(counter)+'.jpg'
            cv2.imwrite(test_img_path+file_name_to, img_rev)        
    # Return revised image
    return imgs_rev
        

def main():
    pass



if __name__ =='__main__':
    main()
