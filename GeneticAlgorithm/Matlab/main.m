%Modified date: 05/07/2019
%Author: Haolin Liu

% === % 

clear;
clc;

% main function %
directory = 'dataset_revise';
pop_loadPath = 'data_dict.mat';
file_start_num = 500;

file_category = fullfile(directory, '*.dat');
file_list = dir(file_category);

keySet = {};
valueSet = {};
% data_dict = {};

for i=1:length(file_list)
    name_parts_cell = split(file_list(i).name, '.');
    name_part = name_parts_cell(1);
    file_path = strcat(directory, '\', file_list(i).name);
    data_temp = airfoilItp(file_path);
    keySet{end+1} = int2str(i+file_start_num);
    valueSet{end+1} = data_temp;
end

data_dict = containers.Map(keySet, valueSet);
save(pop_loadPath, '-Map', data_dict);

GA = GeneticAlgorithm;

GA.cross_rate = 0.5; % The probability of crossover;
GA.mutate_rate = 0.01; % The probability of mutation;
GA.n_generations = 150; % The number of generations;
GA.file_start_num = file_start_num;
GA.pop_loadPath = pop_loadPath;
getFundParams(GA);
pickForBezier(GA);
% pop_Bezier = fitForBezier(GA);

for i=1:GA.n_generations
    fitness_vector = fitnessFunc(GA);
    pop_new = select(GA, fitness_vector);
    CrossOver(GA, pop_new);
    [best_fitness, best_index] = max(fitness_vector);
    GA.pop = pop_new;
end

