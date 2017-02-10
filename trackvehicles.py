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
from helperfunctions import get_hog_features

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
from sklearn import tree


# Define global variables
# frame/image objects
num_frames_to_keep = 5  # number of frames to store
recent_hot_windows = [] # list of hot windows identified on recent frames
# classifier and training related objects
clf = None              # classifier object
X_scaler = None         # scaler object for normalizing inputs
# hyper parameters for feature extraction
color_space = 'BGR'     # color space of the images
orient = 8              # HOG orientations
pix_per_cell = 8        # HOG pixels per cell
cell_per_block = 2      # HOG cells per block
hog_channel = 'HSV_ALL' # Can be 'B', 'G', 'R', 'H', 'S', 'V', 'RGB_ALL' or 'HSV_ALL'
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32          # Number of histogram bins
spatial_feat = False     # Spatial features on or off
hist_feat_RGB = False    # Histogram features on or off on RGB image
hist_feat_HSV = False   # Histogram features on or off on HSV image
hog_feat = True         # HOG features on or off   
# Threshold for procesing heatmaps
thresh=10
# Search area coordinates and window sizes for far, mid-range and near cars
search_window_0 = (np.array([[0.0,1.0], [0.5, 1.0]]), 32)
search_window_1 = (np.array([[0.0,1.0], [0.5, 1.0]]), 64)
search_window_2 = (np.array([[0.0,1.0], [0.5, 1.0]]), 96)
search_window_3 = (np.array([[0.0,1.0], [0.5, 1.0]]), 128)
all_search_windows = [search_window_0,
                      search_window_1, 
                      search_window_2,
                      search_window_3]
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
    global clf    
    global X_scaler

    
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
                        hist_feat_RGB=hist_feat_RGB, 
                        hist_feat_HSV=hist_feat_HSV, 
                        hog_feat=hog_feat)
    
    
    t1=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t1-t0, 2), 'Seconds to extract features from vehicle_features_trn...')
    non_vehicle_features_trn = extract_features(non_vehicles_trn, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat_RGB=hist_feat_RGB, 
                        hist_feat_HSV=hist_feat_HSV, 
                        hog_feat=hog_feat)    
    

    t2=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t2-t1, 2), 'Seconds to extract features from non_vehicle_features_trn...')
    vehicle_features_tst = extract_features(vehicles_tst, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat_RGB=hist_feat_RGB, 
                        hist_feat_HSV=hist_feat_HSV, 
                        hog_feat=hog_feat)
    
    t3=time.time()
    # if verbose, print some details
    if verbose:
        print(round(t3-t2, 2), 'Seconds to extract features from vehicle_features_tst...')
    non_vehicle_features_tst = extract_features(non_vehicles_tst, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat_RGB=hist_feat_RGB, 
                        hist_feat_HSV=hist_feat_HSV, 
                        hog_feat=hog_feat)    
    
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
        print('Using:',orient,'orientations, ',
              pix_per_cell,'pixels per cell, ', 
              cell_per_block,'cells per block, and hog_channel = ', hog_channel)
        print('Feature vector length:', len(scaled_X_trn[0]))

    t0=time.time()
    # Use a Random Forest Classifier
    #clf = RandomForestClassifier(n_estimators=5, max_features=50, 
    #                             max_depth=3, min_samples_split=100, verbose=0)
    clf = LinearSVC()    
    clf.fit(scaled_X_trn, y_trn)

    t1 = time.time()
    
    # if verbose, print some details
    if verbose:
        print(round(t1-t0, 2), 'Seconds to train classifier...')
        # Check the score of the classifier
        print('Test Accuracy of the classifier = ', round(clf.score(scaled_X_tst, y_tst), 4))

    
    t_finish = time.time()
    
    # if verbose, print some details    
    if verbose:
        print('Total time: ', round(t_finish-t_start, 2), 'Seconds')
    
    

def draw_bboxes_using_watershed(img, heatmap, color=(0,0,255), thick=2, verbose=False):
    '''
    Draw bounding boxes around the cars identified in labels heatmap
    img: original image
    heatmap: the heatmap that is created from on_windows
    this algorithm uses watershed algorithm to put bounding boxes around cars that are close together separately
    '''
    # Create masked image from the thresholded heatmap - make image single channel
    masked_img = np.copy(heatmap[:,:,0])
    masked_img[masked_img>0]=1
    # Calculate distance from the background
    distance = ndi.distance_transform_edt(masked_img)
    # Calculate local maxima and convert into dtype=int
    local_maxima = peak_local_max(distance, indices=False, footprint=np.ones((96,96)))
    local_maxima = local_maxima.astype(int)
    # Use label function to identiry and label various local maxima that is found
    markers = label(local_maxima)[0]
    # Use watershed algorithm to identify various portions of the image and assume each is a car
    labels = watershed(-distance, markers, mask=masked_img)
    # Identify the number of cars found
    n_cars = labels.max()
    # if verbose, print some details    
    if verbose:
        print(n_cars, ' cars found')
        cv2.imwrite(test_img_path+'pipeline_4.jpg', labels*100)
    # Iterate through all detected cars
    for car_number in range(1, n_cars+1):
        # Find pixels with each car_number label value
        nonzero = (labels == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img



def draw_bboxes_using_label(img, heatmap, color=(0,0,255), thick=2, verbose=False):
    '''
    Draw bounding boxes around the cars identified in labels heatmap
    img: original image
    labels: result of the label fundction
    '''
    # Create labels from the heatmap
    labels = label(heatmap)
    # if verbose, print some details    
    if verbose:
        print(labels[1], ' cars found')
        print('maximum intensity of heatmap = ', heatmap.max())
        cv2.imwrite(test_img_path+'pipeline_4.jpg', labels[0]*100)
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




def mark_vehicles_on_frame(frame_img, verbose=False, plot_heat_map=False, plot_box=True, watershed=False):
    '''
    Identify the vehicles in a frame and return the revised frame with vehicles identified
    with bounding boxes
    '''
    
    # Define global variables
    global recent_hot_windows
    
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
        hot_windows += search_windows(frame_img, search_window, slide_windows, clf, X_scaler, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat_RGB=hist_feat_RGB, 
                                hist_feat_HSV=hist_feat_HSV, 
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
    # if verbose, plot the heatmap and save to file   
    if verbose:
        scaled_heatmap = heatmap*100
        scaled_heatmap[scaled_heatmap>255] = 255
        scaled_heatmap[:,:,:2] = 0
        cv2.imwrite(test_img_path+'pipeline_2.jpg', cv2.addWeighted(np.copy(frame_img), 1, scaled_heatmap, 0.5, 0))
    
    # Zero out pixels below the threshold
    heatmap[heatmap <= thresh] = 0    
    # if verbose, save some photos and print some details   
    if verbose:
        print('maximum intensity of heatmap = ', heatmap.max())
        scaled_heatmap = heatmap*100
        scaled_heatmap[scaled_heatmap>255] = 255
        scaled_heatmap[:,:,:2] = 0
        cv2.imwrite(test_img_path+'pipeline_3.jpg', cv2.addWeighted(np.copy(frame_img), 1, scaled_heatmap, 0.5, 0))
    
    # Draw the bounding boxes on the images
    draw_image = np.copy(frame_img)
    if plot_box:
        draw_color = [0,0,0]
        draw_color[color_space.index('B')] = 255
        draw_color = tuple(draw_color)
        if watershed:
            draw_image = draw_bboxes_using_watershed(draw_image, heatmap, color=draw_color, thick=1, verbose=verbose) 
        else:
            draw_image = draw_bboxes_using_label(draw_image, heatmap, color=draw_color, thick=1, verbose=verbose) 
    if plot_heat_map:
        scaled_heatmap = heatmap*100
        scaled_heatmap[scaled_heatmap>255] = 255
        scaled_heatmap[:,:,:2] = 0
        draw_image = cv2.addWeighted(draw_image, 1, scaled_heatmap, 0.5, 0)
    
    return draw_image



def process_movie(file_name, pre_fix='AK_', threshold=10, c_space='RGB'):
    '''
    Load movie and replace frames with processed images and then save movie back to file
    '''
    global thresh
    global color_space
    thresh = threshold 
    color_space = c_space
    
    movie_clip = VideoFileClip(work_path+file_name)
    processed_clip = movie_clip.fl_image(mark_vehicles_on_frame)
    processed_clip.write_videofile(work_path+pre_fix+file_name, audio=False, verbose=True, threads=6)




def process_test_images(sequence=False, verbose=False, threshold=4, watershed=False):
    '''
    Read test images, process them, mark the vehicles on them and save them back to the folder
    '''
    global recent_hot_windows
    global thresh
    
    thresh = threshold
    
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
            img_rev = mark_vehicles_on_frame(img, verbose=verbose, watershed=watershed)
            # Recorde time and print details if verbose = True
            if verbose:
                t_finish = time.time()
                print('Total time for ', os.path.basename(file_name_from), ': ', round(t_finish-t_start, 2), 'Seconds')
            # save image
            # Save image to file
            file_name_to = 'processed_'+os.path.basename(file_name_from)
            cv2.imwrite(test_img_path+file_name_to, img_rev) 

    

def read_data_and_train_classifier(limit_trn=-1, random=True):
    '''
    reset clf and X_scaler variables, read training data and train the classifier from scratch
    '''
    global clf
    global X_scaler
    clf = None
    X_scaler = None
    print('reading datasets')
    v_trn, v_tst, nv_trn, nv_tst = read_datasets(limit_trn=limit_trn, random=random)
    train_classifier(v_trn[0], v_tst[0], nv_trn[0], nv_tst[0], verbose=True)
    return v_trn, v_tst, nv_trn, nv_tst



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



def plot_hog_images(data_tuple, num_images=5, cs_list=['RGB','HSV','YCrCb','LUV','Gray']):
    '''
    plots random hog features images from the dataset that is passed on to this function.
    data_tupe: 2-tuple with first element containing the images, and 2nd element containnig the filenames.
    num_images: number of images to choose from the dataset
    cs_list: list of color spaces
    '''
    # creating the grid space
    hspace = 0.2    # distance between images vertically
    wspace = 0.01   # distance between images horizontally
    n_rows = num_images
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


def extract_g():
    i_tree = 0
    for tree_in_forest in clf.estimators_:
        with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
        i_tree = i_tree + 1


def main():
    pass


if __name__ =='__main__':
    main()
