% Given air foil coordinates, call xfoil and return
% lift drag ratio
function [ratio] = fitness(x)
    try
        [pol, ~] = xfoil(x, 0, 4e5, 0, 'oper/iter 500');
    catch
        ratio=0;
    end
    ratio=pol.CL/pol.CD;
end

