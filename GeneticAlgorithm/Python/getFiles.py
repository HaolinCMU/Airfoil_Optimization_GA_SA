# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:54:56 2019

@author: haolinl
"""

import os
import shutil

def moveFiles(directory_namesource, directory_source, targetFolder):
    '''
    Move files into target folder. 
    Parameters:
        directory_namesource: the path of folder that contains '.npy' files, the name source of files
        directory_source: the path of folder that contains to-be-moved files 
        targetFolder: the path of target folder
    '''
    path_list = os.listdir(directory_namesource)
    name_list = []
    for path in path_list:
        real_path = directory_namesource + '/' + path
        if (os.path.isdir(real_path)): continue
        file_name = path.split('.')[0]
        name_list.append(file_name)
    
    check_list = os.listdir(directory_source)
    for file in name_list:
        check_name = file + '.dat'
        if check_name not in check_list: continue 
        fileName = directory_source + '/' + file + '.dat'
        shutil.move(fileName, targetFolder)

if __name__ == "__main__":
#    fund_string = 'D:/CMU/Courses/2019Spring/24-785A：Engineering Optimization/project/dataset'
#    directory_namesource = fund_string + '/dataset_new'
#    targetFolder = fund_string + '/dataset_source'
#    directory_source = fund_string + '/dataset_source/finished'
    
    
    directory_namesource = 'dataset_new'
    targetFolder = 'dataset_revise'
    directory_source = 'dataset_revise/finished'
    

    moveFiles(directory_namesource, directory_source, targetFolder) # Move back

__name__ = "__main__"
