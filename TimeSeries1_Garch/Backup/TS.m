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

for sumofdiff = 1:50
%minimise using fminunc; x(1) = omega; x(2)=alpha; x(3)=beta; x(4) = d
  hT_1;
  eta(1);
  bgarch = @(x)((ones(1,length(hT_1)))*(log(2*x(1) + 2*x(2)*hT_1.*eta.^2 + x(3)*hT_1) + ((z1plus - x(4)*znolag).^2)./(2*x(1) + 2*x(2)*hT_1.*eta.^2 + x(3)*hT_1)));
  x0 = rand(1,4);
  [x,fval,info, output, grad, hess] = fminunc(bgarch,x0);
  hT_1 = [2*x(1) + (2*x(2)*hT_1.*eta.^2 + x(3)*hT_1)];
  eta = (z1plus-x(4)*znolag)./hT_1;
  %sumofdiff = sum(abs(hT_1 - h));
  endfor

omega_bgarch = x(1)
alpha_bgarch = x(2)
beta_bgarch = x(3)
delta_bgarch = x(4)
var_bgarch = (diag(hess^(-1)))'
t_val_bgarch = x./var_bgarch
%bgarch model finished


###########################################################################################################################
