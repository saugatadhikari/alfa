import torch
import os
import re
import numpy as np
import math
from matplotlib import pyplot as plt
import sys

sys.path.append("../")
import config
from config import *

import shutil

from tqdm import tqdm


def pad_data(unpadded_data):
    
    height = unpadded_data.shape[0]
    width = unpadded_data.shape[1]
    
    width_multiplier = math.ceil(width/SPATIAL_SIZE)
    height_multiplier = math.ceil(height/SPATIAL_SIZE)
    
    new_width = SPATIAL_SIZE*width_multiplier
    new_height = SPATIAL_SIZE*height_multiplier
    
    width_pad = new_width-width
    height_pad = new_height-height
        
    if width_pad%2 == 0:
        left = int(width_pad/2)
        right = int(width_pad/2)
    else:
        # print("Odd Width")
        left = math.floor(width_pad/2)
        right = left+1
    
    if height_pad%2 == 0:
        top = int(height_pad/2)
        bottom = int(height_pad/2)
    else:
        # print("Odd Height")
        top = math.floor(height_pad/2)
        bottom = top+1
    
    data_padded = np.pad(unpadded_data, pad_width = ((top, bottom), (left, right)), mode = 'reflect')
        
    assert data_padded.shape[0]%SPATIAL_SIZE == 0, f"Padded height must be multiple of SPATIAL_SIZE: {SPATIAL_SIZE}"
    assert data_padded.shape[1]%SPATIAL_SIZE == 0, f"Padded width must be multiple of SPATIAL_SIZE: {SPATIAL_SIZE}"
    
    return data_padded


def crop_data(uncropped_data, region_num):
    
    output_path = "./cropped_al"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    height = uncropped_data.shape[0]
    width = uncropped_data.shape[1]
    
    # print("crop input height: ", height)
    # print("crop input width: ", width)
    
    vertial_patches = height//SPATIAL_SIZE
    horizontal_patches = width//SPATIAL_SIZE
    
    # print("vertial_patches: ", vertial_patches)
    # print("horizontal_patches: ", horizontal_patches)
    # print(region_num)
    
    cropped_data = []
    
    for y in range(0, vertial_patches):
        for x in range(0, horizontal_patches):
            new_name = f"Region_{region_num}"+"_y_"+str(y)+"_x_"+str(x)+"_label.npy"
            
            # print("new_name: ", new_name)
            
            x_start = (x)*SPATIAL_SIZE
            x_end = (x+1)*SPATIAL_SIZE
            
            y_start = (y)*SPATIAL_SIZE
            y_end = (y+1)*SPATIAL_SIZE
            
            patch = uncropped_data[y_start: y_end, x_start:x_end]
            
            # print(patch.shape)
            
            np.save(os.path.join(output_path, new_name), patch)

def crop_data_al(uncropped_data, filename, TEST_REGION, is_feature = False):
    # base_path = "./data_al/"
    # output_path = base_path + "cropped_al"

    output_path = os.path.join(os.getcwd(), "data_al/cropped_al")

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    height = uncropped_data.shape[0]
    width = uncropped_data.shape[1]
    
    # print("crop input height: ", height)
    # print("crop input width: ", width)
    
    vertial_patches = height//SPATIAL_SIZE
    horizontal_patches = width//SPATIAL_SIZE
    
    # print("vertial_patches: ", vertial_patches)
    # print("horizontal_patches: ", horizontal_patches)
    # print(filename)
    
    cropped_data = []
    
    for y in range(0, vertial_patches):
        for x in range(0, horizontal_patches):
            
            if is_feature:
                # new_name = filename[:8]+"_y_"+str(y)+"_x_"+str(x)+"_features.npy"
                new_name = f"Region_{TEST_REGION}" + "_y_"+str(y)+"_x_"+str(x)+"_features.npy"
            else:
                # new_name = filename[:8]+"_y_"+str(y)+"_x_"+str(x)+"_label.npy"
                new_name = f"Region_{TEST_REGION}" + "_y_"+str(y)+"_x_"+str(x)+"_label.npy"
            
            # print("new_name: ", new_name)
            
            x_start = (x)*SPATIAL_SIZE
            x_end = (x+1)*SPATIAL_SIZE
            
            y_start = (y)*SPATIAL_SIZE
            y_end = (y+1)*SPATIAL_SIZE
            
            patch = uncropped_data[y_start: y_end, x_start:x_end]
            
            # print(patch.shape)
            
            np.save(os.path.join(output_path, new_name), patch)

            
def make_data(label_data, region_num):
    label_file = f"Region_{region_num}_labels.npy"
    
    # ###########Padd data to fit SPATIAL_SIZE pathches######################################
    padded_label = pad_data(label_data)

    ###########Crop data to SPATIAL_SIZE pathches######################################
    crop_data_al(padded_label, label_file, region_num)


def make_dir(TEST_REGION):
    
    if not os.path.exists(f"./Region_{TEST_REGION}_TEST"):
        os.mkdir(f"./Region_{TEST_REGION}_TEST")
        os.mkdir(f"./Region_{TEST_REGION}_TEST/cropped_data_val_test_al")
        
    if not os.path.exists(f"./Region_{TEST_REGION}_TEST/cropped_data_val_test_al"):
        os.mkdir(f"./Region_{TEST_REGION}_TEST/cropped_data_val_test_al")    
    
        
def move_files(TEST_REGION):
    # base_path = "./data_al/"
    base_path = os.path.join(os.getcwd(), "data_al/")
    
    for file in tqdm(os.listdir(base_path + "cropped_al")):
        if not "label" in file:
            continue
            
        file_region_num = int(file.split("_")[1])
        if file_region_num != TEST_REGION:
            continue

        source = os.path.join(base_path + "cropped_al", file)
        
        destination = os.path.join(base_path + f"Region_{TEST_REGION}_TEST/cropped_data_val_test_al", file)
        shutil.move(source, destination)
        

def remake_data(label_data, region_num):
    
    ## Make directories for train_test regions 
    make_dir(region_num)
    
    ## Pad and crop data
    make_data(label_data, region_num)
    
    ## Move image crops to directory
    move_files(region_num)
    

# if __name__ == "__main__":
#     main()
