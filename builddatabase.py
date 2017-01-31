# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:00:19 2017

@author: ali.khalili
"""

import glob
import os
from shutil import copyfile
import pandas as pd
import cv2
import numpy as np


# path to the working repository
work_path = 'C:/Users/ali.khalili/Desktop/Car-ND/CarND-P5-Vehicle-Detection-and-Tracking/'
# path to the final non-vehicle dataset of 64 x 64 images
non_vehicle_path = work_path + 'non-vehicles-dataset-final/'
# path to the final vehicle dataset of 64 x 64 images
vehicle_path = work_path + 'vehicles-dataset-final/'

# Source folder pahts for GTI and KITTI vehicle datasets
GTI_path1 = work_path + 'vehicles-udacity/vehicles/GTI_Far/'
GTI_path2 = work_path + 'vehicles-udacity/vehicles/GTI_Left/'
GTI_path3 = work_path + 'vehicles-udacity/vehicles/GTI_MiddleClose/'
GTI_path4 = work_path + 'vehicles-udacity/vehicles/GTI_Right/'
KITTI_path = work_path + 'vehicles-udacity/vehicles/KITTI_extracted/'

# Source folder pahts for GTI and Extra non-vehicle datasets
GTI_path5 = work_path + 'non-vehicles-udacity/non-vehicles/GTI/'
Extras_path = work_path + 'non-vehicles-udacity/non-vehicles/Extras/'

# Soruce folder path fro AUTTI dataset
AUTTI_path = work_path + 'object-dataset-autti/'

# Soruce folder path fro Crowdai dataset
CrowdAI_path = work_path + 'object-detection-crowdai/'


def delete_current_datasets(verbose=True):
    '''
    Delete all files in the current dataset of vehicle and non-vehicle images
    '''
    
    # Verbose mode
    if verbose:
        print('Deleting the current contents of the database.')
        
    # Make a list of all files in the vehicle dataset folder and delete each file
    file_names = glob.glob(vehicle_path+'*.*')
    for file_name in file_names:
        os.remove(file_name)

    # Make a list of all files in the non-vehicle dataset folder and delete each file
    file_names = glob.glob(non_vehicle_path+'*.*')
    for file_name in file_names:
        os.remove(file_name)
        
        
        

def copy_files(from_dir, to_dir, format_list, verbose=True):
    '''
    Copy files from vehicle datasets of GTI
    '''
    
    # Verbose mode
    if verbose:
        print('Copying from dataset files {} to {}'.format(from_dir, to_dir))
        
    # Iterate for each file format e.g. *.jpg or *.png, etc.
    for file_format in format_list:
        # Obtain all filenames that match the file_format and are in the from_dir        
        file_names_from = glob.glob(from_dir+file_format)
        # Iterate for each filename that isin the from_dir and matches teh file_format
        for file_name_from in file_names_from:
            # Extract the filename from the path
            file_name_to = to_dir+os.path.basename(file_name_from)
            # Copy file to the destination folder
            copyfile(file_name_from, file_name_to)
            


def copy_GTI_dataset(verbose=True):
    '''
    Copy files from the GTI dataset to the final dataset location
    '''
    # Copy the files to the vehicle dataset  
    copy_files(GTI_path1, vehicle_path, ['*.png','*.jpg'], verbose=verbose)
    copy_files(GTI_path2, vehicle_path, ['*.png','*.jpg'], verbose=verbose)
    copy_files(GTI_path3, vehicle_path, ['*.png','*.jpg'], verbose=verbose)
    copy_files(GTI_path4, vehicle_path, ['*.png','*.jpg'], verbose=verbose)
    # Copy the files to the non-vehidle dataset
    copy_files(GTI_path5, non_vehicle_path, ['*.png','*.jpg'], verbose=verbose)
    


def copy_KITTI_dataset(verbose=True):
    '''
    Copy files from the KITTI dataset to the fianl dataset location
    '''
    # Copy the files to the vehicle dataset  
    copy_files(KITTI_path, vehicle_path, ['*.png','*.jpg'], verbose=verbose)
    
    

def copy_Extras_dataset(verbose=True):
    '''
    Copy files from the Extras dataset to the fianl dataset location
    '''
    # Copy the files to the non-vehidle dataset
    copy_files(Extras_path, non_vehicle_path, ['*.png','*.jpg'], verbose=verbose)

    

def copy_autti_dataset(csv_filename='labels.csv', verbose=True):
    '''
    Extract and add all images from the autti dataset to the final location
    '''
    
    # Verbose mode
    if verbose:
        print('Extracting and saving images from the AUTTI dataset')

    # Read the cvs file accompanied with the dataset
    df = pd.read_csv(AUTTI_path+csv_filename, delimiter=' ', header=None, names=['file_name', 'x_min', 'y_min', 'x_max', 'y_max', 'occluded', 'label', 'attribute'])
    
    # Iterate through the list of objects in the csv file
    for i in range(len(df)):
        # Extract field values for each row in the cvs file
        file_name = df['file_name'][i]        
        x_min = df['x_min'][i]
        x_max = df['x_max'][i]
        y_min = df['y_min'][i]
        y_max = df['y_max'][i]
        label = df['label'][i]
        # Read the image
        img = cv2.imread(AUTTI_path+file_name)
        # Extract the image related to the object
        img_ext = img[y_min:y_max,x_min:x_max,:]
        # Write the image back
        if label=='car' or label=='truck':
            cv2.imwrite(vehicle_path+str(i)+'.png',img_ext)
        else:
            cv2.imwrite(non_vehicle_path+str(i)+'.png',img_ext)
            
    # Return df for further manipulation if required
    return df



def copy_crowdai_dataset(csv_filename='labels.csv', verbose=True):
    '''
    Extract and add all images from the crowdai dataset to the final location
    '''
    
    # Verbose mode
    if verbose:
        print('Extracting and saving images from the Crowdai dataset')    
    
    # Read the cvs file accompanied with the dataset
    df = pd.read_csv(CrowdAI_path+csv_filename, delimiter=',', header=0)
    # Rename columns - just so taht I can copy and paste the code above 
    df.rename(columns={'Frame':'file_name', 'xmin':'x_min', 'xmax':'x_max', 'ymin':'y_min', 'ymax':'y_max', 'Label':'label'}, inplace=True)
    
    # Iterate through the list of objects in the csv file
    for i in range(len(df)):
        # Extract field values for each row in the cvs file
        file_name = df['file_name'][i]        
        x_min = df['x_min'][i]
        x_max = df['x_max'][i]
        y_min = df['y_min'][i]
        y_max = df['y_max'][i]
        label = df['label'][i]
        # Read the image
        img = cv2.imread(CrowdAI_path+file_name)
        # Extract the image related to the object
        img_ext = img[y_min:y_max,x_min:x_max,:]
        # Write the image back
        if label=='Car' or label=='Truck':
            cv2.imwrite(vehicle_path+str(i)+'.png',img_ext)
        else:
            cv2.imwrite(non_vehicle_path+str(i)+'.png',img_ext)
        
    # Return df for further manipulation if required
    return df



def count_images_in_dataset(verbose=True):
    '''
    Count the number of images in the datasets
    '''
    
    # Verbose mode
    if verbose:
        print('Extracting and saving images from the Crowdai dataset')    
    
    # consider all files in the folder to be images in the dataset.
    file_format = '*.*'
    
    # Obtain all filenames that match the file_format and are in the from_dir        
    vehicle_file_names = glob.glob(vehicle_path+file_format)
    non_vehicle_file_names = glob.glob(non_vehicle_path+file_format)
    
    if verbose:
        print('There are {} files in vehicle dataset and {} files in non-vehicle dataset.'.format( \
        len(vehicle_file_names), len(non_vehicle_file_names)))
    
    return (len(vehicle_file_names), len(non_vehicle_file_names))




def generate_additional_data(target_number=20000, verbose=False):
    '''
    Generate additional data by tweaking images in both vehicle and non-vehicle datasets until the 
    target number of images is available in each dataset.
    '''
    
    # consider all files in the folder to be images in the dataset.
    file_format = '*.*'
    # Obtain all filenames that match the file_format and are in the from_dir        
    vehicle_file_names = glob.glob(vehicle_path+file_format)
    non_vehicle_file_names = glob.glob(non_vehicle_path+file_format)    
    
    # Set transformation parameters
    # Maximum absolute number of pixels for transformation - i.e. 5 means from -5 to 5 pixels
    max_move = 5 
    # Maximum absolute angle of rotation - i.e. 10 meanse from -10 to +10 degrees
    max_rotate = 15 
    # Maximum scaling factor minus 1 in absolute terms - i.e. 0.3 means from 0.7 to 1.3 scaling
    max_scale_diff = 0.2 
    # Maximum move of the (57, 57) point in absolute terms. The other two points of (7, 7), and (7, 57) will not move.
    max_affine = 5 
    
    
    # Calculate number of images currently in the dataset    
    n_vehicle = len(vehicle_file_names)
    # Iterate randomly through existing images, transform them, and add them to the dataset
    for i in range(target_number-n_vehicle):
        
        # Load a random image
        file_name = vehicle_file_names[int(round(np.random.uniform(n_vehicle)))-1]
        img = cv2.imread(file_name)
        
        # extracting image sizes and setting border mode
        num_rows = img.shape[0]
        num_cols = img.shape[1]
        bm = cv2.BORDER_REFLECT
    
        # translating images
        t_x = np.random.uniform()*max_move*2-max_move
        t_y = np.random.uniform()*max_move*2-max_move
        trans_m = np.float32([[1,0,t_x],[0,1,t_y]])
        img = cv2.warpAffine(img,trans_m,(num_cols,num_rows),borderMode=bm)
    
        # rotating images
        t_r = np.random.uniform()*max_rotate*2-max_rotate
        t_s = np.random.uniform()*max_scale_diff*2-max_scale_diff+1
        rot_m = cv2.getRotationMatrix2D((num_cols/2,num_rows/2),t_r,t_s)
        img = cv2.warpAffine(img,rot_m,(num_cols,num_rows),borderMode=bm)
    
        # affine transform
        a_m1 = np.float32([[7,7],[57,57],[7,57]])
        t_a = np.random.uniform()*max_affine*2-max_affine
        a_m2 = np.float32([[7,7],[57+t_a,57+t_a],[7,57]])
        aft_m = cv2.getAffineTransform(a_m1, a_m2)
        img = cv2.warpAffine(img,aft_m,(num_cols,num_rows),borderMode=bm)
        
        # Flip horizontally with a probability of 50%
        if int(round(np.random.uniform(2))) == 1:
            img = cv2.flip(img, 1)
    
        file_name_to = os.path.basename(file_name)
        file_name_to = file_name_to[:file_name_to.find('.')]
        file_name_to = vehicle_path+file_name_to+'_'+str(i)+'.png'
        if verbose:
            print(os.path.basename(file_name_to))
        cv2.imwrite(file_name_to, img)

    # Calculate number of images currently in the dataset        
    n_non_vehicle = len(non_vehicle_file_names)
    # Iterate randomly through existing images, transform them, and add them to the dataset
    for i in range(target_number-n_non_vehicle):
        
        # Load a random image
        file_name = non_vehicle_file_names[int(round(np.random.uniform(n_non_vehicle)))-1]
        img = cv2.imread(file_name)
        
        # extracting image sizes and setting border mode
        num_rows = img.shape[0]
        num_cols = img.shape[1]
        bm = cv2.BORDER_REFLECT
    
        # translating images
        t_x = np.random.uniform()*max_move*2-max_move
        t_y = np.random.uniform()*max_move*2-max_move
        trans_m = np.float32([[1,0,t_x],[0,1,t_y]])
        img = cv2.warpAffine(img,trans_m,(num_cols,num_rows),borderMode=bm)
    
        # rotating images
        t_r = np.random.uniform()*max_rotate*2-max_rotate
        t_s = np.random.uniform()*max_scale_diff*2-max_scale_diff+1
        rot_m = cv2.getRotationMatrix2D((num_cols/2,num_rows/2),t_r,t_s)
        img = cv2.warpAffine(img,rot_m,(num_cols,num_rows),borderMode=bm)
    
        # affine transform
        a_m1 = np.float32([[7,7],[57,57],[7,57]])
        t_a = np.random.uniform()*max_affine*2-max_affine
        a_m2 = np.float32([[7,7],[57+t_a,57+t_a],[7,57]])
        aft_m = cv2.getAffineTransform(a_m1, a_m2)
        img = cv2.warpAffine(img,aft_m,(num_cols,num_rows),borderMode=bm)
        
        # Flip horizontally with a probability of 50%
        if int(round(np.random.uniform(2))) == 1:
            img = cv2.flip(img, 1)
    
        file_name_to = os.path.basename(file_name)
        file_name_to = file_name_to[:file_name_to.find('.')]
        file_name_to = non_vehicle_path+file_name_to+'_'+str(i)+'.png'
        if verbose:
            print(os.path.basename(file_name_to))
        cv2.imwrite(file_name_to, img)




def prepare_and_augment_datasets():
    '''
    Copy, extrace and/or save all relevant files to the dataset locations
    Augment the dataset to contain equal number of images
    '''
    # Delete_current_datasets
    delete_current_datasets(verbose=False)
    # Copy and save images from GTI dataset
    copy_GTI_dataset(verbose=False)
    # Copy and save images from KITTI dataset
    copy_KITTI_dataset(verbose=False)
    # Copy and save images from KITTI dataset
    copy_Extras_dataset(verbose=False)
    # Print the number of images in datasets before augmenting
    count_images_in_dataset()
    # Augment the datasets to contain 20000 images each
    generate_additional_data(target_number=20000, verbose=False)
    # Print the number of images in datasets after augmenting
    count_images_in_dataset()




def main():
    pass
        
        
    

if __name__ == '__main__':
    main()
