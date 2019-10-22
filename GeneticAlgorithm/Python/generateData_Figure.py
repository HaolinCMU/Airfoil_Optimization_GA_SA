# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:30:27 2019

@author: haolinl
"""

"""
This file is used to generate interpolation points and save the coordinate data into figures.  
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import skimage
import skimage.io
from scipy.interpolate import interp1d
import os
import re
import getFiles

def readDatToMat(file_path):
    '''
    Read file and convert strings into float numbers, and then output matrix.
    Remove lines that have the same x value. 
    Make the data monotonic along the contour. 
    Parameters:
        file_path: the relative path of file.
    Return:
        data: A matrix of float number data.
    '''
    with open(file_path, 'rt') as f:
        lines = f.read().splitlines()
    data_list = []
    for line in lines[1:]:
        if (line == '' or line == ' '): continue 
        one_line_data = re.findall(r"[-+]?\d*\.\d+|\d+", line) # Extract all float numbers from the string, return a list
        data_list.append([float(i) for i in one_line_data])
    data = np.array(data_list).reshape(-1,2)
    
    # Remove rows that have the same x value -- only keep one
    x = data[:,0]
    x_min = min(list(x))
    index_min = np.argwhere(x==x_min)
    if index_min.shape[0] != 1:
        data = np.delete(data, list(index_min[1:,0]), 0)
    
    # Make sure the x coordinates change monotonically
    index_min = list(data[:,0]).index(min(list(data[:,0]))) # Find the index of minimum x value
    data_part1 = data[0:index_min+1,:]
    data_part2 = data[index_min+1:,:] # Divide data into two parts from 'index_min'
    # Monotonize data_part1
    for gene in data_part1:
        index_dup = np.argwhere(data_part1[:,0]==gene[0])
        if index_dup.shape[0] == 1: continue 
        data_part1 = np.delete(data_part1, list(index_dup[1:,0]), 0)
    # Monotonize data_part2
    for gene in data_part2:
        index_dup = np.argwhere(data_part2[:,0]==gene[0])
        if index_dup.shape[0] == 1: continue 
        data_part2 = np.delete(data_part2, list(index_dup[1:,0]), 0)
    data = np.vstack((data_part1, data_part2)) # stack two parts of data vertically
    
    return data

def writeFile(data, write_path):
    '''
    Write the data matrix into "write_path". 
    Parameters:
        data: To-be-written data matrix;
        write_path: Path referring to the target file
    '''
    content = []
    for i in range(len(data)):
        line_temp = ' '.join([str(n) for n in data[i,:]])
        content.append(line_temp)
    content = '\n'.join(content)
    with open(write_path, 'w') as f:
        f.write(content)

def plotFig(x_up, y_up, x_down, y_down, xnew_up, ynew_up, xnew_down, ynew_down):
    '''
    Plot the figure of airfoil's interpolated contour
    Parameters:
        x_up: Original x coordinates on the upside of contour
        y_up: Original y coordinates on the upside of contour
        x_down: Original x coordinates on the downside of contour
        y_down: Original y coordinates on the downside of contour
        xnew_up: X coordinates of the new interpolated points on the upside of contour
        ynew_up: Y coordinates of the new interpolated points on the upside of contour
        xnew_down: X coordinates of the new interpolated points on the downside of contour
        ynew_down: Y coordinates of the new interpolated points on the downside of contour
    '''
    fig = pylab.figure()
    pylab.plot(x_up, y_up, 'o', x_down, y_down, 'o', xnew_up, ynew_up, 
             '-', xnew_down, ynew_down, '-')
    fig.show()
    pylab.close(fig)
    
def airfoilItp(file_path, interNum=(300,100), interType='cubic'):
    '''
    Read file from the path and interpolate points to required number. 
    Parameters:
        file_path: path of the source file
        interNum(default): A tuple which contains two interpolation numbers that define the seed density.  # Appear agian at crossover function. 
        interType(default): the type of interpolation
    Return:
        data_new: 'Nx2' matrix containing points and values of interpolation. 
    '''
    data = readDatToMat(file_path)
    length = np.size(data, axis=0)
    min_id = list(data[:,0]).index(min(list(data[:,0]))) # Find the position of the point on the left end of contour
    x_min = data[min_id,0]
    x_max_1 = data[0,0]
    x_max_2 = data[length-1,0] # To define the interpolation range
    
    x_up = data[0:min_id,0]
    y_up = data[0:min_id,1]
    x_down = data[min_id:,0]
    y_down = data[min_id:length,1]
    
    f_up = interp1d(x_up, y_up, kind=interType, fill_value='extrapolate')
    f_down = interp1d(x_down, y_down, kind=interType, fill_value='extrapolate')
    
    # Must assign different interpolation density on different parts (head or tail) of contour. 
    xnew_up_divide = x_min+(x_min+x_max_1)/4
    xnew_up_tail = np.linspace(x_max_1, xnew_up_divide, num=interNum[1]+1)
    xnew_up_head = np.linspace(xnew_up_divide, x_min, num=interNum[0])
    xnew_up = np.hstack((xnew_up_tail[0:interNum[1]], xnew_up_head))
    
    xnew_down_divide = x_min+(x_max_2+x_min)/4
    xnew_down_head = np.linspace(x_min, xnew_down_divide, num=interNum[0]+2)
    xnew_down_tail = np.linspace(xnew_down_divide, x_max_2, num=interNum[1])
    xnew_down = np.hstack((xnew_down_head[1:interNum[0]+1], xnew_down_tail))
    
    ynew_up = f_up(xnew_up)
    ynew_down = f_down(xnew_down)
    
    x_new = np.vstack((xnew_up.reshape(-1,1), xnew_down.reshape(-1,1)))
    y_new = np.vstack((ynew_up.reshape(-1,1), ynew_down.reshape(-1,1)))
    data_new = np.hstack((x_new, y_new)) # Form a new interpolation matrix, whose first column is x value, and second column is y value
    
    return data_new

def genFig(data_matrix, save_contour_path, save_fig_path, fig_size=12.8, resolution=20):
    '''
    Read the data matrix and plot them to generate figure of contour.
    Parameters:
        data_matrix: the matrix containing all data (interpolated)
        save_contour_path: the path of to-be-saved contour figure (in '.png' format)
        save_fig_path: the path of to-be-saved filled-in figure (in '.png' format)
#        save_fig_folder(default, optinal): the path of folder which contains rotated figures. 
        fig_size(default): the size of to-be-generated figure
        resolution(default): the resolution (size of the black pixel block) of filled-in figure
#        alpha_min(default): the minimum (lower limit) of angle of attack
#        alpha_max(default): the maximum (higher limit) of angle of attack
#        alpha_increment(default): the increment size of angle of attack
    Return:
        im0: the binary matrix of image. 
    '''
    # Save the contour
    fig = pylab.figure(figsize=(fig_size, fig_size))
    pylab.plot(data_matrix[:,0], data_matrix[:,1], c='k')
    pylab.xlim(0,1)
    pylab.ylim(-0.4,0.4)
    pylab.axis('off')
    fig.savefig(save_contour_path, dpi=10)
    pylab.close(fig)
    
    # Read the conotur and save the filled-in figure
    img0 = skimage.io.imread(save_contour_path)
    im0 = skimage.color.rgb2gray(skimage.color.rgba2rgb(img0))
    
    # Change the gray figure into black (binary figure)
    for i in range(im0.shape[0]):
        for j in range(im0.shape[1]):
            if im0[i,j] < 1: im0[i,j] = 0
    
    # Map the pixels inside the contour into zero
    for k in range(im0.shape[1]):
        zero_index_temp = np.argwhere(im0[:,k]==0)
        if (zero_index_temp.shape[0] == 0 or zero_index_temp.shape[0] == 1): continue
        im0[zero_index_temp[0,0]:zero_index_temp[len(zero_index_temp)-1,0]+1,k] *= 0
            
    fig = pylab.figure(figsize=(fig_size, fig_size))
    pylab.axis('off')
    skimage.io.imsave(save_fig_path, im0)
    pylab.close(fig)
    
    return im0

def generateData_Figure(directory, targetFolder, contourFolder, figFolder, txtFolder, dictFolder):
    '''
    (Put all above functions together.)
    Generate the interpolation data and corresponding contour and filled-in figure; save into different folders. 
    Parameters:
        directory: the folder that contains the source '.dat' file
        targetFolder: the folder that will be used to save the new interpolated '.npy' file
        contourFolder: the folder that will be used to save contour figures
        figFolder: the folder that will be used to save filled-in figures
        txtFolder: the folder that will be used to save the new interpolated data into '.txt' format
        dictFolder: the folder that will be used to save dictionaries ('.npz' format file)
    '''
    data_dict = {}
    img_dict = {}
    file_paths = os.listdir(directory)
    
    for path_id, path in enumerate(file_paths):
        if (os.path.isdir(directory + '/' + path)): continue 
        name_temp = path.split('.')[0]
        read_path = directory + '/' + path
        data_new_temp = airfoilItp(read_path)
        data_dict[name_temp] = data_new_temp # Create dictionary of data matrices
        
        save_path = targetFolder + '/' + name_temp + '.npy'
        np.save(save_path, data_new_temp) # save the matrix into '.npy' file with the same name
        
        write_path = txtFolder + '/' + name_temp + '.txt'
        writeFile(data_new_temp, write_path) # write interpolated matrix into '.txt' file
        
        # Only for testing the figure generating function
        save_contour_path = contourFolder + '/' + name_temp + '.png'
        save_fig_path = figFolder + '/' + name_temp + '.png'
        im0_temp = genFig(data_new_temp, save_contour_path, save_fig_path)
        img_dict[name_temp] = im0_temp # img_dict: the image dictionary
        
        print("Data generated: No.%d | Figures generated: No.%d | File replaced: No.%d " % (path_id+1, path_id+1, path_id+1))
        getFiles.moveFiles(targetFolder, directory, directory+'/finished') # Move extraction finished files into another folder
        
    img_dict_savePath = 'img_dict.npz'
    data_dict_savePath = 'data_dict.npz'
    np.savez(img_dict_savePath, **img_dict) # Save the image matrices dictionary into '.npz' file
    np.savez(data_dict_savePath, **data_dict) # Save the data matrices dictionary into '.npz' file
