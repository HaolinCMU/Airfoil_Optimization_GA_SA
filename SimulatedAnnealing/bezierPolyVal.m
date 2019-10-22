% Given bezier curve parameters return coordinates of air foil
% p: 7*1 vector
%    p(1) slope of a line;
%    p(2), p(3)-> x coords of upper P1, lower P1;
%    p(4), p(5)-> x, y coords of upper P2
%    p(6), p(7)-> x, y coords of lower P2
% y: n*2 matrix
%    foil coordinates starts from trailing edge top to leading edge,
%    then back to trailing edge bottom
% Air Foil shape is describe by two bezier curves, the upper and lower.
% They share 2 control points which are P0 (0, 0) and P3 (1,0)
% To make it G1 continuous at P0 (0,0), constrain P1 of upper and lower
% bezier curve both lie on y=k(x), where k is the 1st element in vector p

function y=bezierPolyVal(p)
    p1=[p(2); p(1)*p(2)];
    p2=[p(4); p(5)];
    t=0.99:-0.01:0;
    upper=p1*3*(1-t).^2.*t+p2*3*(1-t).*t.^2+[1; 0]* t.^3;
    p1=[p(3); p(1)*p(3)];
    p2=[p(6); p(7)];
    t=0.01:0.01:1;
    lower=p1*3*(1-t).^2.*t+p2*3*(1-t).*t.^2+[1; 0]* t.^3;
    y=[upper' ; lower'];
end