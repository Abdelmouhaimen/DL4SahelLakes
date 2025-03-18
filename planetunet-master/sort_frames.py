#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:10:46 2023

@author: mathilde
"""

#%%
import os
import pandas as pd
import json
import numpy as np
import shutil

#%%
run = '20230314-1935_modif20_rgb_b8b11b12_50steps_100e_alpha05'
frames_infos = '/home/mathilde/cnn/other_files/preprocessed_frames/%s/aa_frames_list.json'%run
frames_path = '/home/mathilde/cnn/other_files/preprocessed_frames/%s/'%run
frames_sorted_path = '/home/mathilde/cnn/other_files/preprocessed_frames_split/%s/'%run
#%%

with open(frames_infos) as json_data:
    data = json.load(json_data)
    validation_frames = np.array([sub + '.tif' for sub in pd.DataFrame(data['validation_frames'])[0].values.astype('str') ])
    training_frames = np.array([sub + '.tif' for sub in pd.DataFrame(data['training_frames'])[0].values.astype('str')])
    testing_frames = np.array([sub + '.tif' for sub in pd.DataFrame(data['testing_frames'])[0].values.astype('str')])
    
    
if not os.path.exists(frames_sorted_path) :
    os.mkdir(frames_sorted_path)

if not os.path.exists(frames_sorted_path+'validation_frames/') :
    os.mkdir(frames_sorted_path+'validation_frames/')

if not os.path.exists(frames_sorted_path+'training_frames/') :
    os.mkdir(frames_sorted_path+'training_frames/')
    
if not os.path.exists(frames_sorted_path+'testing_frames/') :
    os.mkdir(frames_sorted_path+'testing_frames/')


for files in validation_frames :
    file_to_copy = frames_path + files
    paste_file = frames_sorted_path+'validation_frames/'
    shutil.copy(file_to_copy, paste_file)
    
for files in training_frames :
    file_to_copy = frames_path + files
    paste_file = frames_sorted_path+'training_frames/'
    shutil.copy(file_to_copy, paste_file) 
    
for files in testing_frames :
    file_to_copy = frames_path + files
    paste_file = frames_sorted_path+'testing_frames/'
    shutil.copy(file_to_copy, paste_file) 
