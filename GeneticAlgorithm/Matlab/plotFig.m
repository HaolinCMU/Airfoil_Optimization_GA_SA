%Modified date: 05/07/2019
%Author: Haolin Liu

% === % 

clear;
clc;

% For plot only %
generation = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150];
fitness_vector_1 = log([31.2611,31.7960,32.0885,32.0885,32.2662,32.2662,32.2662,32.2662,32.2662,32.2662,32.2662,32.2662,32.2662,32.2662,32.2662,32.2662]).*10;
fitness_vector_2 = log([28.0217,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044,29.5044]).*10;
% fitness_vector_3 = log([25.3726,27.5303,28.6108,29.2206,30.8985,31.0214,31.0214,31.0214,31.0214,31.0214,31.0214,31.0214,31.6598,31.6598,31.6598,31.6598]).*10;
% fitness_vector_4 = log([27.2223,30.8761,31.8285,31.9100,31.9100,31.9100,32.2761,32.5808,32.6854,32.6854,32.6854,32.6854,32.6854,32.6854,32.6854,32.6854]).*10;
% fitness_vector_5 = log([29.7302,29.7302,29.7302,30.9100,30.9100,30.9100,30.9100,30.9100,30.9100,30.9100,30.9100,30.9100,30.9100,30.9100,30.9100,30.9100]).*10;
% fitness_vector_6 = log([30.9559,32.6504,33.7510,33.7890,33.7890,33.7890,33.7890,35.0090,35.0090,35.0090,35.0090,35.0090,35.0090,35.0090,35.0090,35.0090]).*10;
% fitness_vector_7 = log([23.6914,24.4239,24.7898,24.7898,24.7898,24.7898,26.3072,26.3072,26.3072,26.3072,26.3072,26.3072,26.3072,26.3072,26.3072,26.3072]).*10;
fitness_vector_1 = fitness_vector_1./sum(fitness_vector_1).*500;
fitness_vector_2 = fitness_vector_2./sum(fitness_vector_2).*500;
% ratio = zeros(7,1);
% matrix = [fitness_vector_1;fitness_vector_2;fitness_vector_3;fitness_vector_4;fitness_vector_5;fitness_vector_6;fitness_vector_7];
% for i=1:7
%     ratio(i) = (matrix(i,7) - matrix(i,1))/matrix(i,1);
% end
% 
% ratio_avg = sum(ratio)/length(ratio);
% disp(ratio);

LineWidth = 1.5;
figure();
plot(generation, fitness_vector_1, '-r', 'LineWidth', LineWidth);
hold on;
plot(generation, fitness_vector_2, '--b', 'LineWidth', LineWidth);
% hold on;
% plot(generation, fitness_vector_3, '-g', 'LineWidth', LineWidth);
% hold on;
% plot(generation, fitness_vector_4, '-k', 'LineWidth', LineWidth);
% hold on;
% plot(generation, fitness_vector_5, '-m', 'LineWidth', LineWidth);
% hold on;
% plot(generation, fitness_vector_6, '-c', 'LineWidth', LineWidth);
% hold on;
% plot(generation, fitness_vector_7, '-y', 'LineWidth', LineWidth);


xlabel('generations');
ylabel('L/D ratio');
title('Lift-to-drag ratio vs. Generations');
legend('without Local Search', 'with Local Search', 'Location', 'best');
% lgd.NumColumns = 2;


