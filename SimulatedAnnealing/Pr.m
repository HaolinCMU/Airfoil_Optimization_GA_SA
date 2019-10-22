function [y] = Pr(T,fNew, f)
    y=exp(-(fNew-f)/T);
end

