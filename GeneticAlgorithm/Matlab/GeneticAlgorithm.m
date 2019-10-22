%Modified date: 05/07/2019
%Author: Haolin Liu

% === % 

classdef GeneticAlgorithm
    properties
%         Initial parameter definition. 
%         Parameters: 
%             cross_rate: the probability of crossover between two chromosomes
%             mutate_rate: the probability of mutation of one gene
%             pop_npzPath: the path that refers to the '.npz' file of data matrices
%             n_generations: the number of generations (iterations)
        
        cross_rate
        mutate_rate
        pop_loadPath % should be a path referring to .mat file which saves a dictionary(map containers)
        n_generations
        file_start_num
        pop = {}
%         pop_pack = {}
        pop_size = 0
        genes_size = 0
%         pop_pack = load(obj.pop_loadPath)
%         pop_size = size(pop_pack,3)        
        gene_width = 2
        tournament_size = 20
        fitness_scalar = 100
        mutation_alter_ratio = 0.001
        errHold = 1e-4
        changeHold = 1e-2
%         file_start_num = 400
%         pop = []
%         for i=1:obj.pop_size
%             pop(end+1) = obj.pop_pack(int2str(i+file_start_num))
%         end
    end
    methods
        function getFundParams(obj)
            % Get several fundamental parameters that need to be derived
            % from property parameters
            
            pop_pack = load(obj.pop_loadPath);
            obj.pop_size = size(pop_pack); % Need to revise
%             pop = {};
            for i=1:obj.pop_size
                obj.pop{end+1} = pop_pack(int2str(i+obj.file_start_num));
            end
            obj.genes_size = size(obj.pop{1},1);
        end
        
        function pickForBezier(obj)
%             Only for Bezier: pick out controlling points;
%         Parameters:
%             control_points_size(default): the number of control points number on one side (upside/downside) of the contour. 
%           
            obj.getFundParams();
            control_points_size = 100;
            half_size = int(obj.genes_size/2);
            up_start = 1;
            up_end = half_size;
            down_start = half_size + 1;
            down_end = obj.genes_size;
            
            index_list_up = [];
            index_list_down = [];
            index_up = linspace(up_start, up_end, control_points_size);
            index_down = linspace(down_start, down_end, control_points_size);
            for i=1:control_points_size
                index_list_up(end+1) = fix(index_up(i));
                index_list_down(end+1) = fix(index_down(i));
            end
            
            for i=1:obj.pop_size
                pop_up_temp = obj.pop{i}(index_list_up,:);
                pop_down_temp = obj.pop{i}(index_list_down,:);
                pop_temp = [pop_up_temp;pop_down_temp];
                obj.pop{i} = pop_temp; % Change the interpolation points to control points (genes)
            end
            
            obj.genes_size = size(obj.pop{1},1);
        end
        
        function pop_Bezier = fitForBezier(obj)
%             Fit the Bezier curve and output a new list with fitted points coordinates.
%         Returns:
%             pop_Bezier: a list that contains the corresponding Bezier fitting points (with same size as original cubic fitting data). 
            
            pop_Bezier = {};
            divide_index = fix(obj.genes_size/2);
            
            for i=1:obj.pop_size
                pop_up_temp = obj.pop{i}(1:divide_index,:);
                pop_up_Bezier = BezierFitting(pop_up_temp);
                
                pop_down_temp = obj.pop{i}(divide_index+1:end,:);
                pop_down_Bezier = BezierFitting(pop_down_temp);
                
                pop_Bezier_temp = [pop_up_Bezier;pop_down_Bezier];
                pop_Bezier{end+1} = pop_Bezier_temp;
            end
        end
        
        function fitness_vector = fitnessFunc(obj)
%             Call pre-trained CNN as fitness function. 
%         Parameters:
%             neural_network: the pre-trained convolutional neural network
%             device: the "cpu" or "cuda" information
%             contourFolder: the path of folder that contains figures of airfoil contour
%             figFolder: the path of folder that contains figures of filled-in airfoil
%         Return:
%             fitness: the fitness value of data (airfoils)
            
            fitness_vector = zeros(1, obj.pop_size);
            pop_Bezier = obj.fitForBezier();
            
            for i=1:obj.pop_size
                % Should import xfoil here. 
                [pol,~] = xfoil(pop_Bezier{i},0,4e5,0.0);
                value_temp = pol.CL/pol.CD;
                fitness_vector(i) = exp(obj.fitness_scalar.*value_temp);
            end
        end
        
        function Ltd = localSearchFunc(obj, control_array)
            Bezier_matrix = BezierFitting(control_array);
            [pol,~] = xfoil(Bezier_matrix,0,4e5,0.0);
            Ltd = -obj.fitness_scalar*(pol.CL/pol.CD);
        end
        
        function localSearch(obj, fitness_vector)
            maxiter = 100;
            [best_fitness, best_index] = max(fitness_vector);
            control_vector_0 = obj.pop{best_index};
            control_vector_0 = reshape(control_vector_0,1,numel(control_vector_0));
            % Conduct nelder mead
        end
        
        function pop_new = select(obj, fitness_vector)
%             Strategy 2: Deterministic tournament selection. 
%         Parameters:
%             fitness_vector: the vector of fitness value that will be used for determine the individual probability of selection
%         Return:
%             pop_new: the selected new population that contains parents with good fitness value. 
            
            pop_new = {};
            for i=1:obj.pop_size
                candidates_index = randsample(obj.pop_size, obj.tournament_size);
                select_fitness = fitness_vector(candidates_index);
                [max,max_index] = max(select_fitness);
                max_index = candidates_index(max_index);
                pop_new{end+1} = obj.pop{max_index};
            end
        end
        
        function satisfy = constraint(obj, chromo, index)
%             Set geometric constraint for airfoil's top head part (convex constraint): 
%             If the point is located at the top head part of contour, then it will be subjected to certain constraint.
%         Parameters:
%             chromo: the current chromo that is going to be mutated. 
%             index: the current index of gene in one chromo. 
%         Returns:
%             satisfied(boolean): True/False that indicates whether or not the mutation satisfies the constriant
        
            startIndex = fix(obj.genes_size/8);
            endIndex = fix(obj.genes_size/2) + fix(obj.genes_size/8);
            
            index_list = linspace(startIndex, endIndex, (endIndex-startIndex+1));
            if ismember(index,index_list)
                slope_1 = (chromo(index-1,1) - chromo(index,1))/(chromo(index-1,2)-chromo(index,2));
                slope_2 = (chromo(index,1) - chromo(index+1,1))/(chromo(index,2)-chromo(index+1,2));
                if slope_1 <= slope_2
                    satisfy = true;
                else
                    satisfy = false;
                end
            else
                if chromo(index,1) > 1 || chromo(index,1) < 0
                    satisfy = false;
                else
                    satisfy = true;
                end
            end
        end
        
        function mutateONE(obj, chromo)
%             Conduct mutation on a single chromo. 
%         Mutation will be smoothed on the chosen point and its neighbor points. 
%         Parameters:
%             chromo: the one to-be-mutated chromo
%             norm_num: the number used to normalize the mutation range

            norm_num = 100;
%             gaussian_range = 5;
%             gaussian_sigma = 1; % Cancel gaussian smoothing on mutation. 
            
            x_range = max(chromo(:,1)-min(chromo(:,1)));
            h_range = (max(chromo(:,1))-min(chromo(:,2)))/2;
            dimension_range = [x_range, h_range];
            chromo_copy = chromo;
            
            fix_point_index_list = [0, fix((obj.genes_size/2)-1), fix(obj.genes_size/2), fix((obj.genes_size/2)+1), obj.genes_size];
            for i=1:obj.genes_size
                if rand() > obj.mutate_rate || ismember(i, fix_point_index_list)
                    continue;
                end
                
                while(true)
                    mutate_value_x = randi([-norm_num, norm_num])/norm_num;
                    mutate_value_y = randi([-norm_num, norm_num])/norm_num;
                    chromo_copy(i,1) = mutate_value_x*obj.mutation_alter_ratio*dimension_range(1);
                    chromo_copy(1,2) = mutate_value_y*obj.mutation_alter_ratio*dimension_range(2);
                    if obj.constraint(chromo_copy,i)
                        break
                    end 
                end
                
%                 if (i - fix(gaussian_range/2) < 0 || i + fix((gaussian_range/2)+1) > obj.genes_size)
                chromo(i,1) = chromo(i,1) + mutate_value_x*obj.mutation_alter_ratio*dimension_range(1);
                chromo(i,2) = chromo(i,2) + mutate_value_y*obj.mutation_alter_ratio*dimension_range(2);
%                 else
%                     mutate_vector = zeros(2, gaussian_range);
%                     mutate_vector(1, fix(gaussian_range/2)) = mutate_value_x;
%                     mutate_vector(2, fix(gaussian_range/2)) = mutate_valeu_y;
%                     mutate_vector(1,:) = [];
%                 end
            end
        end
        
        function CrossOver(obj, pop_new)
%             Conduct crossover (mating) between different chromosomes in one population pool and generate offsprings; Mutate with mutation rate on each contour
%         Only swap the corresponding upside or downside part of airfoil contour (for better match)
%         Parameters:
%             pop_new: take in new population (parents)
            
            cross_pts_index = fix(obj.genes_size/2) - 1;
            pop_copy = pop_new;
            
            for i=1:obj.pop_size
                if rand() > obj.cross_rate
                    continue;
                end
                chosen_index = randsample(obj.pop_size, 1);
                dice = rand();
                if dice >= 0.5
                    pop_new{i}(1:cross_pts_index+1,:) = pop_copy{chosen_index}(0:cross_pts_index+1,:);
                else
                    pop_new{i}(cross_pts_index+2,:) = pop_copy{chosen_index}(cross_pts_index+2,:);
                end
                obj.mutateONE(pop_new{i});
            end
        end
        
        function judge = criteria(obj, best_fitness_array)
%              Define the algorithm's criteria.
%         Parameters;
%             best_fitness_array: the array that stores the best values in continuous number of iterations
%         Return: 
%             (Boolean): A boolean value that indicates whether or not the algorithm reaches its threshold. 
%                 False: continue iterating; True: stop iterating before next generation. 
            
            mse = SquareLoss(best_fitness_array);
            if (mse/obj.fitness_scalar <= obj.errHold)
                judge = 'converge';
            elseif (mse/obj.fitness_scalar <= obj.changeHold && mse/obj.fitness_scalar > obj.errHold)
                judge = 'convert';
            else
                judge = 'continue';
            end
        end
    end
end