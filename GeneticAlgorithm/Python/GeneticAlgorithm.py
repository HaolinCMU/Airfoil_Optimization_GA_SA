# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:24:33 2019

@author: haolinl
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import random
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import generateData_Figure as GDF
import scipy.ndimage
import scipy.special
import copy
from cnn import Net1
from scipy.optimize import minimize

def BezierFitForOne(data_matrix, fit_point_num=400):
    '''
    Confuct Bezier curve fitting for one data matrix (airfoil)
    Parameters:
        data_matrix: matrix for only one airfoil
        fit_point_num(default): the points number for curve fitting (on each side)
    Returns:
        data_matrix_fitted: the matrix of fitting points. 
    '''
    divide_index = int(data_matrix.shape[0]/2)
    Bezier_up = BezierFitting(data_matrix[0:divide_index,:], fit_point_num)
    Bezier_down = BezierFitting(data_matrix[divide_index:,:], fit_point_num)
    
    data_matrix_fitted_up = Bezier_up.fitting()
    data_matrix_fitted_down = Bezier_down.fitting()
    
    data_matrix_fitted = np.vstack((data_matrix_fitted_up, data_matrix_fitted_down))
    
    return data_matrix_fitted


def plotBestFig(matrix_list, num, fig_size=12.8, string='result'):
    '''
    PLot the matrix into contour figure.
    Parameters:
        matrix: the matrix containing points coordinates on airfoil's contour
        num: the number of 'int((gen+1)/10)'
        fig_size(default): the size of figure
        string(default): the path for saving the final result
    '''
    save_path = string + '/' + str(num) + '.png'
    fig = pylab.figure(figsize=(fig_size, fig_size))
    pylab.plot(matrix_list[0][:,0], matrix_list[0][:,1], 'b-', 
               matrix_list[1][:,0], matrix_list[1][:,1], 'g--')
    pylab.xlim(0,1)
    pylab.ylim(-0.2,0.2)
    fig.savefig(save_path)
    pylab.close(fig)

def SquareLoss(array):
    '''
    Calculate the square loss to mean value for the array. 
    Parameters: 
        array: the 1d array to be calculated. 
    Returns: 
        mse: the square loss of the array
    '''
    mean = np.mean(array)
    mse = np.sum([(array[i]-mean)**2 for i in range(len(array))]) / len(array)
    
    return mse

class GeneticAlgorithm(object):
    '''
    The class for genetic algorithm
    '''
    def __init__(self, cross_rate, mutate_rate, pop_npzPath, n_generations):
        '''
        Initial parameter definition. 
        Parameters: 
            cross_rate: the probability of crossover between two chromosomes
            mutate_rate: the probability of mutation of one gene
            pop_npzPath: the path that refers to the '.npz' file of data matrices
            n_generations: the number of generations (iterations)
        '''
        self.gene_width = 2 # the dimension of coordinates of one gene (one interpolation point)
        self.tournament_size = 20 # tournament_size: the number of candidates in one tournament. (Only for deterministic tournament selection)
        self.fitness_scalar = 100 # fitness_scalar: the scalar of fitness value (distribution coefficient)
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.pop_npzPath = pop_npzPath
        self.n_generations = n_generations
        self.mutation_alter_ratio = 0.001 # alter_ratio: the range of mutation on one single gene (x&y coordinates) on one chromo # initial num: 0.01
        self.swap_num = 8 # swap_num: the number of parts of chromosome to be swapped. # For strategy 1 only. # initial num: 8 (abandoned)
        self.pop_npz = np.load(self.pop_npzPath)
        self.pop_size = np.size(self.pop_npz) # pop_size: the number of airfoils (population)
        self.pop = []
        file_start_num = 451
        for i in range(self.pop_size):
            self.pop.append(self.pop_npz[str(i+file_start_num)])
        self.genes_size = np.size(self.pop[0], axis=0) # genes_size: the number of genes on one chromosome
        self.errHold = 1e-4 # errHold: the error tolerance of this criteria. 
        self.changeHold = 1e-2 # changeHold: the changing threshold for 'mutate_ratio' and 'mutation_alter_ratio'        
    
    def pickForBezier(self, control_points_size=100): # If using Bezier curve fitting, this function must be called after generating first batch of data (cubic interpolating)
        '''
        Only for Bezier: pick out controlling points;
        Parameters:
            control_points_size(default): the number of control points number on one side (upside/downside) of the contour. 
        '''
        half_size = int(self.genes_size/2)
        up_start = 0
        up_end = half_size - 1
        down_start = half_size
        down_end = self.genes_size - 1
        
        index_list_up = [int(np.linspace(up_start, up_end, control_points_size)[i]) for i in range(control_points_size)]
        index_list_down = [int(np.linspace(down_start, down_end, control_points_size)[i]) for i in range(control_points_size)]
        
        for i in range(self.pop_size):
            pop_up_temp = self.pop[i][index_list_up,:]
            pop_down_temp = self.pop[i][index_list_down,:]
            pop_temp = np.vstack((pop_up_temp, pop_down_temp))
            self.pop[i] = pop_temp
        self.genes_size = np.size(self.pop[0], axis=0)
        
    def fitForBezier(self):
        '''
        Fit the Bezier curve and output a new list with fitted points coordinates.
        Returns:
            pop_Bezier: a list that contains the corresponding Bezier fitting points (with same size as original cubic fitting data). 
        '''
        pop_Bezier = []
        divide_index = int(self.genes_size/2)
        for i in range(self.pop_size):
            pop_up_temp = self.pop[i][0:divide_index,:]
            Bezier_up = BezierFitting(pop_up_temp)
            pop_up_Bezier = Bezier_up.fitting()
            
            pop_down_temp = self.pop[i][divide_index:,:]
            Bezier_down = BezierFitting(pop_down_temp)
            pop_down_Bezier = Bezier_down.fitting()
            
            pop_Bezier_temp = np.vstack((pop_up_Bezier, pop_down_Bezier))
            pop_Bezier.append(pop_Bezier_temp)
            
        return pop_Bezier
    
    def fitForBayesian(self):
        '''
        For Bayesian: fit the Bayesian curve and reset 'self.pop'.
        '''
        pass
        
    def fitnessFunc(self, neural_network, device, contourFolder, figFolder): # fitness_scalar initial num: 10
        '''
        Call pre-trained CNN as fitness function. 
        Parameters:
            neural_network: the pre-trained convolutional neural network
            device: the "cpu" or "cuda" information
            contourFolder: the path of folder that contains figures of airfoil contour
            figFolder: the path of folder that contains figures of filled-in airfoil
        Return:
            fitness: the fitness value of data (airfoils)
        '''
        fitness_vector = np.empty(shape=self.pop_size)
        pop_Bezier = self.fitForBezier() # Bezier curve fitting
        for i in range(self.pop_size):
            save_contour_path = contourFolder + '/' + str(i+1) + '.png'
            save_fig_path = figFolder + '/' + str(i+1) + '.png'
            img_temp = GDF.genFig(pop_Bezier[i], save_contour_path, save_fig_path).reshape(-1,1) # Bezier
            img_temp = torch.from_numpy(img_temp).float()
            DataLoader_temp = torch.utils.data.TensorDataset(img_temp)
            img_matrix_temp = torch.autograd.Variable(DataLoader_temp[:][0]) # Need to double check
            img_matrix_temp = img_matrix_temp.reshape(-1,1,128,128)
            img_matrix_temp = img_matrix_temp.to(device)
            value_temp = neural_network(img_matrix_temp)
            fitness_vector[i] = np.exp(self.fitness_scalar*value_temp.cpu().data.numpy()[0,0]) # fitness_scalar is called to change the distribution of fitness function
        
        return fitness_vector
    
    def localSearchFunc(self, control_array):
        '''
        Define the function for local search (the function that read control points input and return lift-to-drag ratio)
        Parameters:
            control_array: the matrix that contains coordinates of control points. 
        Return: 
            Ltd: the corresponding lift-to-drag ratio. 
        '''
        # Create a new neural network as fitness function
        cnn_path = 'CNN.pkl'
        contour_path = 'localSearch/contour.png'
        figure_path = 'localSearch/figure.png'
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        neural_network = Net1().to(device)
#        neural_network.load_state_dict(torch.load(cnn_path, map_location='cpu'))
        neural_network.load_state_dict(torch.load(cnn_path)) # For psc
        
        control_matrix = control_array.reshape(-1,2)
        data_matrix = BezierFitForOne(control_matrix)
        img_temp = GDF.genFig(data_matrix, contour_path, figure_path).reshape(-1,1)
        img_temp = torch.from_numpy(img_temp).float()
        DataLoader_temp = torch.utils.data.TensorDataset(img_temp)
        img_matrix_temp = torch.autograd.Variable(DataLoader_temp[:][0])
        img_matrix_temp = img_matrix_temp.reshape(-1,1,128,128)
        img_matrix_temp = img_matrix_temp.to(device)
        value_temp = neural_network(img_matrix_temp)
        Ltd = -self.fitness_scalar*value_temp.cpu().data.numpy()[0,0] # Should be negative for convenience of minimization
        
        return Ltd
        
    def localSearch(self, fitness_vector, maxiter=100):
        '''
        Conduct local searching for reducing time reaching convergence. 
        Using Nelder-Mead. 
        Steps: 
            1. Find the airfoil with the best fitness value
            2. Conduct Nelder-Mead local searching
            3. Replace fitness value and local "best" candidate with the original best one
        Parameters:
            fitness_vector: the vector containing fitness values
            maxiter(default): the maximum number of local search iteration. 
        '''
        pop_copy = copy.deepcopy(self.pop)
        best_index = np.argmax(fitness_vector)
        control_vector_0 = pop_copy[best_index].reshape(1,-1)
        control_candidate = minimize(self.localSearchFunc, control_vector_0, method='nelder-mead', 
                                     options={'maxiter': maxiter})
        fitness_vector[best_index] = -self.localSearchFunc(control_candidate['x'])
        self.pop[best_index] = control_candidate['x'].reshape(-1,2)
    
    def select(self, fitness_vector):
        '''
        Strategy 2: Deterministic tournament selection. 
        Parameters:
            fitness_vector: the vector of fitness value that will be used for determine the individual probability of selection
        Return:
            pop_new: the selected new population that contains parents with good fitness value. 
        '''
        
        # Strategy 2: Tournament selection
        pop_new = []
        for i in range(self.pop_size):
            candidates_index = np.random.choice(np.arange(self.pop_size), size=self.tournament_size, replace=True)
            select_fitness = [fitness_vector[j] for j in candidates_index]
            max_index = candidates_index[np.argmax(select_fitness)]
            pop_new.append(self.pop[max_index])
        return pop_new
    
    def constraint(self, chromo, index):
        '''
        Set geometric constraint for airfoil's top head part (convex constraint): 
            If the point is located at the top head part of contour, then it will be subjected to certain constraint.
        Parameters:
            chromo: the current chromo that is going to be mutated. 
            index: the current index of gene in one chromo. 
        Returns:
            satisfied(boolean): True/False that indicates whether or not the mutation satisfies the constriant
        '''
        startIndex = int(self.genes_size/(2*4))
        endIndex = int(self.genes_size/2) + int(self.genes_size/(2*4)) # Half size plus one third of bottom head part of contour
        
        index_list = list(np.arange(startIndex, endIndex))
        if index not in index_list:
            if (chromo[index,0] > 1 or chromo[index,0] < 0): return False # To see if the x coordinate falls into range [0,1]
            else: return True
        else:
            slope_1 = (chromo[index-1,0] - chromo[index,0])/(chromo[index-1,1] - chromo[index,1])
            slope_2 = (chromo[index,0] - chromo[index+1,0])/(chromo[index,1] - chromo[index+1,1])
            if slope_1 <= slope_2: return True
            else: return False
            
    def mutateONE(self, chromo, norm_num=100, gaussian_range=5, gaussian_sigma=1):
        '''
        Conduct mutation on a single chromo. 
        Mutation will be smoothed on the chosen point and its neighbor points. 
        Parameters:
            chromo: the one to-be-mutated chromo
            norm_num: the number used to normalize the mutation range
            gaussian_range(default): the range of mutating vector for gaussian smoothing; must be odd number. 
            gaussian_sigma(default): the sigma of 1d gaussian filter
        '''
        x_range = max(chromo[:,0]) - min(chromo[:,0]) # The range of x coordinate mutation
        h_range = (max(chromo[:,1]) - min(chromo[:,1]))/2 # The range of y coordinate mutation
        dimension_range = [x_range, h_range] # A list containing two coordinates mutation ranges
        chromo_copy = copy.deepcopy(chromo)
        
        # Conducting mutation only on y coordinate
        fix_point_index_list = [0, int((self.genes_size/2)-1), int(self.genes_size/2), 
                                int((self.genes_size/2)+1), self.genes_size-1]
        for i in range(self.genes_size):
            if (np.random.rand() > self.mutate_rate or (i in fix_point_index_list)): continue
            
            # Check if the mutation satisfies the constraint
            while(True): # To see if the mutation satisfies the constraint
                mutate_value_x = np.random.randint(-norm_num,norm_num)/norm_num # For x
                mutate_value_y = np.random.randint(-norm_num,norm_num)/norm_num # For y
                chromo_copy[i,0] += mutate_value_x*self.mutation_alter_ratio*dimension_range[0]
                chromo_copy[i,1] += mutate_value_y*self.mutation_alter_ratio*dimension_range[1]
                if (self.constraint(chromo_copy, i)): break
            
            if (i-int(gaussian_range/2) < 0 or i+int(gaussian_range/2)+1 > self.genes_size): # In case if the mutate_vector reaches the end of chromo
                chromo[i,0] += mutate_value_x*self.mutation_alter_ratio*dimension_range[0]
                chromo[i,1] += mutate_value_y*self.mutation_alter_ratio*dimension_range[1]
            else: # For the situations that mutate_vector doesn't reach chromo's end
                mutate_vector = np.zeros(shape=(2, gaussian_range))
                mutate_vector[0, int(gaussian_range/2)] = mutate_value_x
                mutate_vector[1, int(gaussian_range/2)] = mutate_value_y
                mutate_vector[0,:] = scipy.ndimage.gaussian_filter1d(mutate_vector[0,:], gaussian_sigma) # Smmoth the mutate_vector, to avoid noisy mutation
                mutate_vector[1,:] = scipy.ndimage.gaussian_filter1d(mutate_vector[1,:], gaussian_sigma)
                chromo[i-int(gaussian_range/2):i+int(gaussian_range/2)+1,0] += mutate_vector[0,:]*self.mutation_alter_ratio*dimension_range[0]
                chromo[i-int(gaussian_range/2):i+int(gaussian_range/2)+1,1] += mutate_vector[1,:]*self.mutation_alter_ratio*dimension_range[1] # Map the mutation to the center point and also nearby points
    
    def CrossOver(self, pop_new):
        '''
        Conduct crossover (mating) between different chromosomes in one population pool and generate offsprings; Mutate with mutation rate on each contour
        Only swap the corresponding upside or downside part of airfoil contour (for better match)
        Parameters:
            pop_new: take in new population (parents)
        '''
        
        # Strategy 2: Binary swapping (need bigger mutating rate. Fix the cross point). 
        cross_pts_index = int(self.genes_size/2) - 1 # 0-399: upper contour; 400-799: lower contour (depend on the sampling points number set in airfoilItp function)
        pop_copy = pop_new.copy()
        
        for i in range(self.pop_size):
            if (np.random.rand() > self.cross_rate): continue
            # When the probability falls into the range of crossover: 
            chosen_index = np.random.randint(self.pop_size)
            dice = np.random.rand() # the number deciding which side will be swapped. 
            if (dice >= 0.5): pop_new[i][0:cross_pts_index+1,:] = pop_copy[chosen_index][0:cross_pts_index+1,:] # swapping upper contour
            else: pop_new[i][cross_pts_index+1:,:] = pop_copy[chosen_index][cross_pts_index+1:,:] # swapping lower contour
            self.mutateONE(pop_new[i]) # Conduct mutation on each parent sample
    
    def criteria(self, best_fitness_array):
        '''
        Define the algorithm's criteria.
        Parameters;
            best_fitness_array: the array that stores the best values in continuous number of iterations
        Return: 
            (Boolean): A boolean value that indicates whether or not the algorithm reaches its threshold. 
                False: continue iterating; True: stop iterating before next generation. 
        '''
        mse = SquareLoss(best_fitness_array)
        if ((mse/self.fitness_scalar) <= self.errHold): return "converge"
        elif ((mse/self.fitness_scalar) <= self.changeHold and (mse/self.fitness_scalar) > self.errHold):
            return "convert"
        else: return "continue"


class BezierFitting(object):
    '''
    Smooth the noisy curve into Bezier curve (with (75, 25) points distribution on each side) for only one side.
    '''
    def __init__(self, data_matrix, fit_point_num=400):
        '''
        Initialize parameters.
        Parameters:
            data_matrix: the matrix of data that are going to be fitted (controlling points) (only one side)
            fit_point_num(default): the number of points on Bezier curve. 
        '''
        self.data_matrix = data_matrix
        self.fit_point_num = fit_point_num
        
    def fitting(self):
        '''
        Conduct the Bezier curve fitting with matrix of controlling points.
        Returns:
            data_fitted: a matrix containing the new ditted data from Bezier curve.
        '''
        size_N = self.data_matrix.shape[0] # The size of control points number
        width = self.data_matrix.shape[1] # The number of dimensions of point coordinates
        order_n = size_N - 1 # The largest order of Bezier curve fitting
        
        binom_coeff = np.array([scipy.special.binom(order_n, i) for i in range(size_N)]).T
        order_t = np.linspace(0, order_n, size_N).T # The orders of term "t" in order
        order_1_t = np.linspace(order_n, 0, size_N).T # # The orders of term "1-t" in order
        
        data_fitted = np.zeros(shape=(self.fit_point_num, width))
        
        # Calculate coordinates of each new fitted point by cross-multiplying coefficient and term vectors
        for i in range(self.fit_point_num):
            t = i / self.fit_point_num
            terms_1_t = np.array((1-t)**order_1_t).T
            terms_t = np.array(t**order_t).T
            x_value_temp = np.sum(binom_coeff*terms_1_t*terms_t*self.data_matrix[:,0])
            y_value_temp = np.sum(binom_coeff*terms_1_t*terms_t*self.data_matrix[:,1])
            
            data_fitted[i,:] = [x_value_temp, y_value_temp]
        
        return data_fitted
