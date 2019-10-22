%Modified date: 05/07/2019
%Author: Haolin Liu

% === % 

function Bezier_data = BezierFitting(control_matrix)
% Smooth the noisy curve into Bezier curve (with (75, 25) points distribution on each side) for only one side.
% Initialize parameters.
% Parameters:
%     data_matrix: the matrix of data that are going to be fitted (controlling points) (only one side)
%     fit_point_num(default): the number of points on Bezier curve. 

fit_point_num = 400;
size_N = size(control_matrix,1);
width = size(control_matrix,2);
order_n = size_N - 1;

binom_coeff = [];
for i=1:size_N
    b_coeff = nchoosek(order_n, i-1);
    binom_coeff(end+1) = b_coeff;
end

order_t = linspace(0, order_n, size_N)';
order_1_t = linspace(order_n, 0, size_N)';

Bezier_data = zeros(fit_point_num, width);
for i=1:fit_point_num
    t = i/fit_point_num;
    terms_1_t = ((1-t).^order_1_t)';
    terms_t = (t.^order_t)';
    x_value_temp = sum(binom_coeff.*terms_1_t.*terms_t.*control_matrix(:,1));
    y_value_temp = sum(binom_coeff.*terms_1_t.*terms_t.*control_matrix(:,2));
    
    Bezier_data(i,:) = [x_value_temp, y_value_temp];
end

end