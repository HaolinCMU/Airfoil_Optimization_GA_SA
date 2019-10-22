clear;
clc;
load foilChange.mat;

for i=1:9
    figure;
    hold on;
    old=bezierPolyVal(foilRes(:,i));
    new=bezierPolyVal(foilRes(:,i+1));
    plot(old(:,1), old(:,2),'r','LineWidth',1);
    plot(new(:,1), new(:,2),'b','LineWidth',1);
    xlabel('x');
    ylabel('y');
    ipr=(ratioRes(i,2)-ratioRes(i,1))/ratioRes(i,1);
    title(['Relative Ratio Improvement: ' num2str(ipr*100) '%']);
    set(gcf, 'Position',  [100, 100, 1000, 400])
    legend('Old Foil', 'New Foil');
    saveas(gcf, [num2str(i) '.png']);
end