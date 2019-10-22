%Modified date: 05/07/2019
%Author: Haolin Liu

% === % 

function writeFile(data, write_path)
% Write the data matrix into "write_path". 
% Parameters:
%     data: To-be-written data matrix;
%     write_path: Path referring to the target file

file = fopen(write_path, 'wt');
for i=1:length(data)
    fprintf(file, '%d %d', data(i,1), data(i,2));
    fprintf(file, '\n');
end
end