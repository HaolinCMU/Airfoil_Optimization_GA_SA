%Modified date: 05/07/2019
%Author: Haolin Liu

% === % 

function data = readDatToMat(file_path)
% Read file and convert strings into float numbers, and then output matrix.
% Remove lines that have the same x value. 
% Make the data monotonic along the contour. 
% Parameters:
%     file_path: the relative path of file.
% Return:
%     data: A matrix of float number data.

data = dlmread(file_path);
data = data(2:end,:);

x = data(:,1);
x_min = min(x);
x_min_index = find(x==x_min);
if size(x_min_index,1) > 1
    data(x_min_index(2:end),:) = [];
end

x_min_index = find(x==min(data(:,1)));
data_part1 = data(1:x_min_index,:);
data_part2 = data(x_min_index+1:end,:);

for i=1:length(data_part1)
    index_dup = find(data_part1(:,1)==data_part1(i,1));
    if size(index_dup,1) > 1
        data_part1(index_dup(2:end),:) = [];
    end
end

for i=1:length(data_part2)
    index_dup = find(data_part2(:,1)==data_part2(i,1));
    if size(index_dup,1) > 1
        data_part2(index_dup(2:end),:) = [];
    end
end

data = [data_part1; data_part2];

end
