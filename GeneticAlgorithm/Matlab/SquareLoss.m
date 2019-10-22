%Modified date: 05/07/2019
%Author: Haolin Liu

% === % 

function mse = SquareLoss(array)
% Calculate the square loss to mean value for the array. 
% Parameters: 
%     array: the 1d array to be calculated. 
% Returns: 
%     mse: the square loss of the array


mean = mean(array);
mse = sum((array - mean).^2);

end