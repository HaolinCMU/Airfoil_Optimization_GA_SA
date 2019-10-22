# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:09:31 2019

@author: haolinl
"""

import os
import shutil
import getFiles

fund_string = 'D:/CMU/Courses/2019Spring/24-785A：Engineering Optimization/project/dataset'

directory_namesource = 'dataset_new'
targetFolder = 'dataset_revise'
directory_source = 'dataset_revise/finished'

getFiles.moveFiles(directory_namesource, directory_source, targetFolder)

target_contour = 'contour'
target_figure = 'figure'
target_txt = 'txt'
target_npy = directory_namesource
img_dict_path = 'img_dict.npz'
data_dict_path = 'data_dict.npz'
optim_dict_path = 'optim_dict.npz'
log_path = 'log.txt'
log_name = '401_450.txt'
result_savePath = log_name.split('.')[0] + '.npz'

folder_list = [target_contour, target_figure, target_npy, target_txt]
path_list = [img_dict_path, data_dict_path, optim_dict_path, log_path, log_name, result_savePath]

# Remove all the folders
for folder in folder_list:
    shutil.rmtree(folder)
    os.mkdir(folder)

# Remove all '.npz' files and log file
for path in path_list:
    if(os.path.exists(path)): os.remove(path)
