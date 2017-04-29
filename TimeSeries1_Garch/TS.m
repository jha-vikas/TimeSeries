clear all

z = csvread("Data.csv");

%prompt = ['p:'];
%dlg_title = 'Please specify the lags i.e. p';
%defaultans = {'1'};
%p = str2double(cell2mat(inputdlg(prompt,dlg_title,1,defaultans)));

%prompt = ['q:'];
%dlg_title = 'Please specify the lags i.e. q';
%defaultans = {'1'};
%q = str2double(cell2mat(inputdlg(prompt,dlg_title,1,defaultans)));

%Get initial values of stochastic errors using OLS
T = length (z); %Use OLS to get the inefficent, but consistent estimates
znolag = z(1:T-1,:);
z1plus = z(2:T,:);
z2plus = z(3:T,:);
[d, sigma, e0] = ols (z1plus, znolag);
e0sq = e0.^2;
omega = alpha = beta = 0.5;
almostzero = 0.000000001;
%h0 = ones(length(z)-1,1)./4;
h0 = 10*rand(length(z)-1,1);
eta = sqrt(e0sq./h0);
h = h0;
########################################################################################################################

%%%%for b garch model
%update the value of h according to bgarch formula
hT_1=h(1:(length(h)),:); 

sumofdiff = 0;

while abs(sum(eta)) > 1
%minimise using fminunc; x(1) = omega; x(2)=alpha; x(3)=beta; x(4) = d
  hT_1;
  eta(1);
  bgarch = @(x)((ones(1,length(hT_1)))*(log(2*x(1) + 2*x(2)*hT_1.*eta.^2 + x(3)*hT_1) + ((z1plus - x(4)*znolag).^2)./(2*x(1) + 2*x(2)*hT_1.*eta.^2 + x(3)*hT_1)));
  x0 = rand(1,4);
  [x,fval,info, output, grad, hess] = fminunc(bgarch,x0);
  hT_1 = [2*x(1) + (2*x(2)*hT_1.*eta.^2 + x(3)*hT_1)];
  eta = (z1plus-x(4)*znolag)./hT_1;
  %sumofdiff = sum(abs(hT_1 - h));
  endwhile

omega_bgarch = x(1)
alpha_bgarch = x(2)
beta_bgarch = x(3)
delta_bgarch = x(4)
var_bgarch = (diag(hess^(-1)))'
t_val_bgarch = x./var_bgarch
%bgarch model finished


###########################################################################################################################
%%%%for t garch model
%update the value of h according to tgarch formula
hT_1=h(1:(length(h)),:); 
lambda = 1;

% hT_1 = (x(1) + x(2)*hT_1.*f(eta) + x(3)*hT_1.^lambda).^(2/lambda)
% f(eta) = ((eta.^2 + almostzero^2).^0.5 - x(5)*eta)
% hT_1 = (x(1) + x(2)*hT_1.*((eta.^2 + almostzero^2).^0.5 - x(4)) + x(3)*hT_1.^lambda).^(2/lambda)

sumofdiff = 0;

eta = sqrt(e0sq./h0);
while abs(sum(eta)) > 1
%minimise using fminunc; x(1) = omega; x(2)=alpha; x(3)=beta; x(4) = d; x(5) = c
  hT_1;
  eta(1);
  tgarch = @(x)((ones(1,length(hT_1)))*(log((x(1) + x(2)*hT_1.*((eta.^2 + almostzero^2).^0.5 - x(4)*eta) + x(3)*hT_1.^lambda).^(2/lambda)) + ((z1plus - x(4)*znolag).^2)./((x(1) + x(2)*hT_1.*((eta.^2 + almostzero^2).^0.5 - x(4)*eta) + x(3)*hT_1.^lambda).^(2/lambda))));
  x0 = rand(1,5);
  [x,fval,info, output, grad, hess] = fminunc(tgarch,x0);
  hT_1 = [(x(1) + x(2)*hT_1.*((eta.^2 + almostzero^2).^0.5 - x(5)*eta) + x(3)*hT_1).^(2)];
  eta = (z1plus-x(4)*znolag)./hT_1;
  %sumofdiff = sum(abs(hT_1 - h));
  endwhile

omega_tgarch = x(1)
alpha_tgarch = x(2)
beta_tgarch = x(3)
delta_tgarch = x(4)
c_tgarch = x(5)
var_tgarch = (diag(hess^(-1)))'
t_val_tgarch = x./var_tgarch

%tgarch model finished

#######################################################################################################
%%%%for GJR garch model
%update the value of h according to GJRgarch formula

hT_1=h(1:(length(h)),:); 
lambda = 2;

% hT_1 = (x(1) + x(2)*hT_1.*f(eta) + x(3)*hT_1.^lambda).^(2/lambda)
% f(eta) = ((1+x(5)^2)*(eta.^2) - x(5)*eta.*(eta.^2 + almostzero^2).^0.5)
% hT_1 = (x(1) + x(2)*hT_1.*((1+x(5)^2)*(eta.^2) - x(5)*eta.*(eta.^2 + almostzero^2).^0.5) + x(3)*hT_1.^lambda).^(2/lambda)
sumofdiff = 0;

eta = sqrt(e0sq./h0);
while abs(sum(eta)) > 1
%minimise using fminunc; x(1) = omega; x(2)=alpha; x(3)=beta; x(4) = d; x(5) = c
  hT_1;
  eta(1);
  GJRgarch = @(x)((ones(1,length(hT_1)))*(log((x(1) + x(2)*hT_1.*((1+x(5)^2)*(eta.^2) - x(5)*eta.*(eta.^2 + almostzero^2).^0.5) + x(3)*hT_1.^lambda).^(2/lambda)) + ((z1plus - x(4)*znolag).^2)./((x(1) + x(2)*hT_1.*((1+x(5)^2)*(eta.^2) - x(5)*eta.*(eta.^2 + almostzero^2).^0.5) + x(3)*hT_1.^lambda).^(2/lambda))));
  x0 = rand(1,5);
  [x,fval,info, output, grad, hess] = fminunc(GJRgarch,x0);
  hT_1 = [(x(1) + x(2)*hT_1.*((1+x(5)^2)*(eta.^2) - x(5)*eta.*(eta.^2 + almostzero^2).^0.5) + x(3)*hT_1.^lambda).^(2/lambda)];
  eta = (z1plus-x(4)*znolag)./hT_1;
  %sumofdiff = sum(abs(hT_1 - h));
  endwhile

omega_GJR_garch = x(1)
alpha_GJR_garch = x(2)
beta_GJR_garch = x(3)
delta_GJR_garch = x(4)
c_GJR_garch = x(5)
var_GJR_garch = (diag(hess^(-1)))'
t_val_GJR_garch = x./var_GJR_garch

%GJR_garch model finished

#######################################################################################

