clear;
clc;
% 
% k=4;
% xu1=0.03;
% xd1=-0.005;
% xu2=0.8;
% yu2=0.01;
% xd2=0.7;
% yd2=-0.01;

% reps=3;
% kArr=repelem(repelem(4,3), reps);
% xu1Arr=repelem(range(0.03, 3, 0.01), reps);
% xd1Arr=[-0.06, -0.05, -0.04, -0.06, -0.05, -0.04, -0.06, -0.05, -0.04];
% xu2Arr=repelem(repelem(0.8, 3), reps);
% yu2Arr=repelem(repelem(0.01, 3), reps);
% xd2Arr=repelem(repelem(0.7, 3), reps);
% yd2Arr=repelem(repelem(-0.01,3), reps);
p1=[4; 0.03; -0.005; 0.8; 0.01; 0.7; -0.01;];
p2=[4; 0.03; -0.005; 0.9; 0.01; 0.7; -0.01;];
p3=[4.1; 0.03; -0.005; 0.8; 0.01; 0.7; -0.01;];
p4=[4.2; 0.03; -0.005; 0.8; 0.01; 0.7; -0.01;];
p5=[4.2; 0.03; -0.005; 0.9; 0.01; 0.7; -0.01;];
p6=[3.9; 0.03; -0.005; 0.9; 0.01; 0.7; -0.01;];
p7=[3.9; 0.03; -0.005; 0.8; 0.01; 0.7; -0.01;];
p8=[3.9; 0.025; -0.005; 0.8; 0.01; 0.7; -0.01;];
p9=[3.7; 0.03; -0.005; 0.9; 0.01; 0.7; -0.01;];

p=[p1 p2 p3 p4 p5 p6 p7 p8 p9];

N=200;
T=4096;
ratioRes=zeros(9,N);
foilRes=zeros(7,9*2);


maxLen=-1;
for i=1:9
    [pArr, ratioArr]=anneal(p(:,i), N, T);
    maxLen=max(maxLen, length(ratioArr));
    for j=2:length(ratioArr)
        if ratioArr(j)<ratioArr(j-1)
            ratioArr(j)=ratioArr(j-1);
        end
    end
    ratioRes(i,1:length(ratioArr))=ratioArr;
    foilRes(:,i)=pArr(:,1);
    foilRes(:,i+1)=pArr(:,end);
end

figure;
hold on;
ratioRes=padLastNum(ratioRes, maxLen);
for i=1:9
    plot(ratioRes(i,1:maxLen),'LineWidth', 1);
end
xlabel('Iteratio');
ylabel('L/D Ratio');
title('L/D Change on Simulated Annealing Algorithm');


function y=range(center, n, precision)
    y=zeros(1,n);
    mid=ceil(n/2);
    for i=1:n
        y(1,i)=center+precision*(i-mid);
    end
end

function [ratioArr]=padLastNum(ratioArr, maxLen)
    len=size(ratioArr);
    for i=1:len(1)
        for j=1:maxLen
            if ratioArr(i,j)==0
                ratioArr(i,j)=ratioArr(i,j-1);
            end
        end
    end
    
end