%Modified date: 05/07/2019
%Author: Haolin Liu

% === % 

function data_new = airfoilItp(file_path)
% Read file from the path and interpolate points to required number. 
% Parameters:
%     file_path: path of the source file
%     interNum(default): A tuple which contains two interpolation numbers that define the seed density.  # Appear agian at crossover function. 
%     interType(default): the type of interpolation
% Return:
%     data_new: 'Nx2' matrix containing points and values of interpolation. 

interNum = [300,100]; % Set the default number of interpolation points;

data = readDatToMat(file_path);
length = size(data, 1);
[x_min, x_min_index] = min(data(:,1));
x_max_1 = data(1,1);
x_max_2 = data(length,1);

x_up = data(1:x_min_index,1);
y_up = data(1:x_min_index,2);
x_down = data(x_min_index+1:end,1);
y_down = data(x_min_index+1:end,2);

xnew_up_divide = x_min + (x_min+x_max_1)/4;
xnew_up_tail = linspace(x_max_1, xnew_up_divide, interNum(2)+1);
xnew_up_head = linspace(xnew_up_divide, x_min, interNum(1));
xnew_up = [xnew_up_tail(1:interNum(2)), xnew_up_head];

xnew_down_divide = x_min + (x_max_2+x_min)/4;
xnew_down_head = linspace(x_min, xnew_down_divide, interNum(1)+2);
xnew_down_tail = linspace(xnew_down_divide, x_max_2, interNum(2));
xnew_down = [xnew_down_head(2:interNum(2)+1), xnew_down_tail];

ynew_up = interp1(x_up, y_up, xnew_up, 'cubic', 'extrap');
ynew_down = interp1(x_down, y_down, xnew_down, 'cubic', 'extrap');

x_new = [xnew_up';xnew_down'];
y_new = [ynew_up';ynew_down'];

data_new = [x_new, y_new];

end