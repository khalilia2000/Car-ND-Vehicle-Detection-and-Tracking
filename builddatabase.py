# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:00:19 2017

@author: ali.khalili
"""

import glob
import os
from shutil import copyfile


# path to the working repository
work_path = 'C:/Users/ali.khalili/Desktop/Car-ND/CarND-P5-Vehicle-Detection-and-Tracking/'
# path to the final non-vehicle dataset of 64 x 64 images
non_vehicle_path = work_path + 'non-vehicles-dataset-final/'
# path to the final vehicle dataset of 64 x 64 images
vehicle_path = work_path + 'vehicles-dataset-final/'



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
            # Extract the filename fro the path
            file_name_to = to_dir+file_name_from[file_name_from.find("\\")+1:]
            # Copy file to the destination folder
            copyfile(file_name_from, file_name_to)
            



def main():
    # Delete_current_datasets()
    delete_current_datasets()
    
    # Destination folder pahts for GTI and KITTI vehicle datasets
    GTI_dir1 = work_path + 'vehicles-udacity/vehicles/GTI_Far/'
    GTI_dir2 = work_path + 'vehicles-udacity/vehicles/GTI_Left/'
    GTI_dir3 = work_path + 'vehicles-udacity/vehicles/GTI_MiddleClose/'
    GTI_dir4 = work_path + 'vehicles-udacity/vehicles/GTI_Right/'
    GTI_dir5 = work_path + 'vehicles-udacity/vehicles/KITTI_extracted/'
    
    # Copy the files to the vehicle dataset  
    copy_files(GTI_dir1, vehicle_path, ['*.png','*.jpg'])
    copy_files(GTI_dir2, vehicle_path, ['*.png','*.jpg'])
    copy_files(GTI_dir3, vehicle_path, ['*.png','*.jpg'])
    copy_files(GTI_dir4, vehicle_path, ['*.png','*.jpg'])
    copy_files(GTI_dir5, vehicle_path, ['*.png','*.jpg'])
    



if __name__ == '__main__':
    main()