function [pArr, ratioArr]=anneal(p, N, T)
    lr=[1e-2; 1e-3; 1e-4; 1e-2; 1e-3; 1e-2; 1e-3];
    % N=50;
    % T=4096;
    % k=4;
    % xu1=0.03;
    % xd1=-0.005;
    % xu2=0.8;
    % yu2=0.01;
    % xd2=0.7;
    % yd2=-0.01;
    % p=[k; xu1; xd1; xu2; yu2; xd2; yd2;];

    pArr=[p];
    ratioArr=[fitness(bezierPolyVal(p))];
%     drawFoil(bezierPolyVal(p));
    try
        for i=1:N
            pNew=p+lr.*(rand(size(p))-0.5);
            ratioNew=fitness(bezierPolyVal(pNew));
            ratio=fitness(bezierPolyVal(p));
            if (ratioNew > ratio)
                p=pNew;
                pArr=[pArr p];
                
            elseif rand > Pr(T, ratioNew, ratio)
                p=pNew;
                pArr=[pArr p];
            end
            ratioArr=[ratioArr ratioNew];
            T=0.9*T;
        end
    catch
%         figure;
%         plot(ratioArr);
%         xlabel('Iteration Number');
%         ylabel('Lift Drag Ratio');
        return;
    end
end



