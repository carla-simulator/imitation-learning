# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:33:53 2019

@author: musta
"""
def write_data(imags_matrix, targets_matrix, directory, filename):
    with h5py.File(directory + filename, 'w') as hdf:
        hdf.create_dataset('rgb',data=imags_matrix)
        hdf.create_dataset('targets',data=targets_matrix)
    

import numpy as np
import matplotlib as plt 
import h5py
# change these directories to suit up you case of use
data_dir = 'S:/Graduation project/Data/AgentHuman/SeqTrain'                      # Data directory 
fl_dir = 'S:/Graduation project/Data/AgentHuman/SeqTrain/Follow'                 # Follow line data folder directory
right_dir = 'S:/Graduation project/Data/AgentHuman/SeqTrain/Right'               # Right data folder directory
left_dir = 'S:/Graduation project/Data/AgentHuman/SeqTrain/Left'                 # Left data folder directory
st_dir = 'S:/Graduation project/Data/AgentHuman/SeqTrain/Straight'               # Unspecisifed data folder directory 
us_dir = 'S:/Graduation project/Data/AgentHuman/SeqTrain/Unsepecified'           # Straight data folder directory 
rg_dir = 'S:/Graduation project/Data/AgentHuman/SeqTrain/Reach'                  # Reach goal data folder directory

# Initilaizing variables to count up to 200, and to use for file naming 

fl_ct = right_ct = left_ct = st_ct = us_ct = rg_ct = 0
fl_file = right_file = left_file = st_file = us_file = rg_file = 0

check = True

# Data size
size = 200 

for _ in range(3663, 6952):
    filename = data_dir + '/data_0' + str(_) + '.h5'
    with h5py.File(filename, 'r') as hdf:
        data = hdf.get('rgb')
        data = np.array(data[:,:,:])
        targets = hdf.get('targets')
        targets = np.array(targets)
    if(check == True):  # Creating matrices with the same size as the data points 
        im_fl = data
        tar_fl = targets
        im_right = data
        tar_right = targets
        im_left = data
        tar_left = targets
        im_st = data
        tar_st = targets
        im_us = data
        tar_us = targets
        im_rg = data
        tar_rg = targets
        check = False
    
    for i in range (0,size):
        imag, target = data[i], targets[i]
        if(target[24] == 2.0):                 # Follow lane
            im_fl[fl_ct] = imag
            tar_fl[fl_ct] = target
            fl_ct = fl_ct + 1
            if(fl_ct == 200):
                fl_ct = 0
                filename = '/data_0' + str(fl_file) + '.h5'
                write_data(im_fl, tar_fl, fl_dir, filename)
                fl_file = fl_file + 1       
                     
        elif(target[24] == 3.0):               # Left 
            im_left[left_ct] = imag
            tar_left[left_ct] = target
            left_ct = left_ct + 1
            if(left_ct == 200):
                left_ct = 0
                filename = '/data_0' + str(left_file) + '.h5'
                write_data(im_left, tar_left, left_dir, filename)
                left_file = left_file + 1       
            
        elif(target[24] == 4.0):               # Right
            im_right[right_ct] = imag
            tar_right[right_ct] = target
            right_ct = right_ct + 1
            if(right_ct == 200):
                right_ct = 0
                filename = '/data_0' + str(right_file) + '.h5'
                write_data(im_right, tar_right, right_dir, filename)
                right_file = right_file + 1    
                
        elif(target[24] == 5.0):               # Straight  
            im_st[st_ct] = imag
            tar_st[st_ct] = target
            st_ct = st_ct + 1
            if(st_ct == 200):
                st_ct = 0
                filename = '/data_0' + str(st_file) + '.h5'
                write_data(im_st, tar_st, st_dir, filename)
                st_file = st_file + 1                    
        elif(target[24] == 0):               # Reach goal 
            im_rg[rg_ct] = imag
            tar_rg[rg_ct] = target
            rg_ct = rg_ct + 1
            if(rg_ct == 200):
                rg_ct = 0
                filename = '/data_0' + str(rg_file) + '.h5'
                write_data(im_rg, tar_rg, rg_dir, filename)
                rg_file = rg_file + 1                 
        else:                                # uncalssified data
            im_us[us_ct] = imag
            tar_us[us_ct] = target
            us_ct = us_ct + 1
            if(us_ct == 200):
                us_ct = 0
                filename = '/data_0' + str(us_file) + '.h5'
                write_data(im_us, tar_us, us_dir, filename)
                us_file = us_file + 1    
