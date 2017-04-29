%The code is written to run in Octave, which is similar to MatLab, but is available free. With some 
%tweaks, the code can be run in MatLab too.
%Get the data file in csv format. Name the data file is "Data.csv"
%Check that the Data file is in the working directory before running.
%The code uses tsa toolbox, financial toolbox, optim toolbox, econometrics toolbox and struct toolbox.
%Same are available at source forge as freeware.
%The program checks for ARCH, GARCH, E-GARCH
clear all

x = csvread("Data.csv");

%p = input("What is the number of lags of the conditional variance? ")
%FOr further better modeling, p will be given by the user.

%q = input("What is the number of lags of the squared innovations? ")
%For further modeling, q will be given by the user.

p = q = 1; %For the time being, it will work with only 1 unit lag and 1 unit innovations. (1,1) model

printf ("Types of model considered are : ARCH, GARCH, e-GARCH, AGARCH \n")

% (((sigma_t)^lambda)-1)/lambda = omega  + alpha*sigma_t-1 *

T = length (x); %Use OLS to get the inefficent, but consistent estimates
x1 = x(1:T-1,:);
x2 = x(2:T,:);
[b, sigma, e] = ols (x2, x1);
esq = e.^2;


%For the ARCH model
t = length(esq);
esq1 = esq(1:t-1,:);
esq2 = [zeros(length(esq1),1)+1,esq1]; %adding a column of 1s for constant
esq3 = esq(2:t, :);
alpha = ((esq2'*esq2)^-1)*(esq2'*esq3);

for i = 1:100
h = esq2*alpha;
z = [1./h, esq1./h];
f = (esq3./h)-1;
alpha_u = ols(f, z);
alpha = alpha + alpha_u;

h_1 = h(2:length(h),:); %size is 299x1, it gives h(t+1)
h_2 = h(1:length(h)-1,:); %size is 299x1, it gives h(t)
alpha1 = alpha(2:length(alpha),:);
e_1 = e(2:(length(e)-1),:); % error at time t+1, size is 299x1
e_2 = e(1:(length(e)-2),:); %error at time t ; size is 299x1
r_1 = 2*((e_2*alpha1)./h_1).^2;

r = sqrt(1./h_2 + r_1);

s_1 = ((alpha1.*alpha1)./h_1).*(((e_1.*e_1)./h_1)-1);
s = 1./h_2 + s_1;

etilda = (e_2.*s)./r;
xtilda = x(2:length(x)-2).*r;

b_u = ols(etilda, xtilda);

b = b + b_u;

endfor

printf("The value of alpha0 for ARCH(1) model is %f\n", alpha(1,:))
printf("The value of alpha1 for ARCH(1) model is %f\n", alpha(2,:))

printf("The value of beta for ARCH(1) model is %f\n", b)







