# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:01:15 2019

@author: haolinl
"""

"""
Add criteria in algorithm
Try to remove noise for each airfoil's generation (fitting with bezier curve with different order on different parts). 
Add constriant for geometry
Reduce the generation number (limited to no more than 300 iters)
Use tournament selection for "select" function
Reduce the number of points (genes_size)? - By Bezier curve fitting 
Combine with local search (Nelder-Mead)? 
Fix the [0,1] interval as constraint; 
"""

"""
04/24/2019 4th
Start:

Parameters:
    cross_rate = 0.5
    mutate_rate = 0.02
    n_generations = 200
    self.mutation_alter_ratio = 0.001
    fitness_scalar: 100
    Selecting strategy: Normal selection
    Tournament_size: 100
    gaussian_range = 7
    gaussian_sigma = 1
    control points number: 100 + 100 = 200
    fit_point_num: 400
    errHold: 1e-4
    
"""

import GeneticAlgorithm as GenAlg
import generateData_Figure as GDF
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from cnn import Net1
import copy


def saveLog(file_name, GA_object, informLines, date_round):
    '''
    save result of optimization (parameters)
    Parameters:
        file_name: the name of file that are going to be written
        GA_object: the object derived from the class 'GeneticAlgorithm'
        informLines: a list of strings containing iteration infromation
        date_round: a tuple containing a pair of strings which refer to date and round number. 
    '''
    content = [file_name.split('.')[0],
               date_round[0],
               date_round[1],
               "cross_rate = %.2f" % GA_object.cross_rate,
               "mutate_rate = %.3f" % GA_object.mutate_rate,
               "n_generations = %d" % GA_object.n_generations,
               "mutation_alter_ratio = %.3f" % GA_object.mutation_alter_ratio,
               "Tournament size = %d" % GA_object.tournament_size,
               "fitness_scalar = %d" % GA_object.fitness_scalar, 
               "pop_size = %d" % GA_object.pop_size,
               "genes_size = %d" % GA_object.genes_size]
    content += informLines
    content = '\n'.join(content)
    with open(file_name, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    # Paths of directory
    fund_string = 'D:/CMU/Courses/2019Spring/24-785A：Engineering Optimization/project/dataset'
    
    directory = 'dataset_revise'
    targetFolder = 'dataset_new'
    contourFolder = 'contour'
    figFolder = 'figure'
    txtFolder = 'txt'
    dictFolder = fund_string
    pop_npzPath = 'data_dict.npz'
    optim_dict_savePath = 'optim_dict.npz'
    cnn_path = 'CNN.pkl'
    
    log_name = '451_500.txt'
    date_round = ("5/2/2019", "Round 40th")
    
    result_savePath = log_name.split('.')[0] + '.npz'
    
    cross_rate = 0.5
    mutate_rate = 0.01 # initial num: 0.01
    n_generations = 200 # initial num: 150
    optim_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neural_network = Net1().to(device)
#    neural_network.load_state_dict(torch.load(cnn_path, map_location='cpu')) # For local laptop
    neural_network.load_state_dict(torch.load(cnn_path)) # For psc
    
    
    # Generating initial dataset
    GDF.generateData_Figure(directory, targetFolder, contourFolder, figFolder, txtFolder, dictFolder)
    # Create genetic algorithm project
    GA = GenAlg.GeneticAlgorithm(cross_rate, mutate_rate, pop_npzPath, n_generations)
    GA.pickForBezier() # Pick the controlling points for Bezier curve fitting. 
    
    # Set a array for storing best fitness value. 
    best_fitness_length = 5
    best_fitness = np.zeros(best_fitness_length)
    informLines = []
    
    initial_best_fitness = 0
    initial_best_airfoil = None
    global_best_fitness = 0
    global_best_airfoil = None
    generation_list = []
    fitness_list = []
    
    for gen in range(n_generations):
        fitness_vector = GA.fitnessFunc(neural_network, device, contourFolder, figFolder)
#        GA.localSearch(fitness_vector)
        pop_new = GA.select(fitness_vector)
        GA.CrossOver(pop_new)
        best_index = np.argmax(fitness_vector)
        if (gen == 0):
            pop_initial = copy.deepcopy(GA.pop)
            initial_best_fitness = np.log(fitness_vector[best_index])
            initial_best_airfoil = pop_initial[best_index]
            global_best_fitness = initial_best_fitness
            global_best_airfoil = initial_best_airfoil
        else: 
            if (np.log(fitness_vector[best_index]) >= global_best_fitness):
                pop_best = copy.deepcopy(GA.pop)
                global_best_fitness = np.log(fitness_vector[best_index])
                global_best_airfoil = pop_best[best_index]
        GA.pop = pop_new

        # Push newest best fitness value into array, and calculate to see if it meets the criteria
        for i in range(best_fitness_length):
            if (i == best_fitness_length-1): best_fitness[i] = np.log(fitness_vector[best_index])
            else: best_fitness[i] = best_fitness[i+1]

        informLine_temp = ' | '.join(["Gen: %d" % (gen+1),
                                     "best fit: %.4f" % (np.log(fitness_vector[best_index])),
                                     "index: %d" % best_index,
                                     "MSE: %.4f" % GenAlg.SquareLoss(best_fitness)])

        print(informLine_temp)
        informLines.append(informLine_temp)
        saveLog(log_name, GA, informLines, date_round) # Save iteration information after every iteration
        
        if (GA.criteria(best_fitness) == "converge"): 
            print("The algorithm meets its criteria, therefore stopped. ")
            informLine_best = "Best fitness value: %.4f" % global_best_fitness
            print(informLine_best)
            informLines += ["The algorithm meets its criteria, therefore stopped. "]
            informLines += [informLine_best]
            
            initial_bezier_airfoil = GenAlg.BezierFitForOne(initial_best_airfoil)
            best_bezier_airfoil = GenAlg.BezierFitForOne(global_best_airfoil)
            plotList = [initial_bezier_airfoil, best_bezier_airfoil]
            GenAlg.plotBestFig(plotList, int((gen+1)/10))
            for i in range(len(pop_new)): optim_dict[str(i+1)] = pop_new[i]
            break
        # Save reesults every 10 eopchs
        if ((gen+1) % 10 == 0):
            informLine_best = "Best fitness value: %.4f" % global_best_fitness
            print(informLine_best)
            informLines += [informLine_best]
            
            initial_bezier_airfoil = GenAlg.BezierFitForOne(initial_best_airfoil)
            best_bezier_airfoil = GenAlg.BezierFitForOne(global_best_airfoil)
#            print(global_best_airfoil[0:4,:])
            plotList = [initial_bezier_airfoil, best_bezier_airfoil]
            GenAlg.plotBestFig(plotList, int((gen+1)/10))
            for i in range(len(pop_new)): optim_dict[str(i+1)] = pop_new[i]
            # Save the generation num. vs. global_best_fitness
            generation_list.append(gen+1)
            fitness_list.append(global_best_fitness)
        
        #Change mutation rate and alter_ratio
        if ((gen+1) % 50 == 0):
            GA.mutate_rate *= 1.25
            GA.mutation_alter_ratio *= 1.25
    
    # Save the result into a dictionary
    result = {}
    result['Initial_airfoil'] = initial_bezier_airfoil
    result['Best_airfoil'] = best_bezier_airfoil
    result['Initial_fitness'] = initial_best_fitness
    result['Best_fitness'] = global_best_fitness
    result['generation_list'] = generation_list
    result['fitness_list'] = fitness_list
    
    np.savez(result_savePath, **result)
    saveLog(log_name, GA, informLines, date_round) # Save all the iteration information
    np.savez(optim_dict_savePath, **optim_dict)


__name__ = "__main__"
