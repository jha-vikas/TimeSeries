clear all
bgarch = @(x)3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2);
x0 = [1,1];
[x,fval] = fminunc(bgarch,x0);