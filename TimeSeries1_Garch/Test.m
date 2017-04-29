function [y,l] = Test(x)
y = sum(x)/length(x);
l = sqrt(sum((x-y).^2/length(x)));
end
Test2;