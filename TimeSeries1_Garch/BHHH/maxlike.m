function [theta1,f,cov] = maxlike(fhandle,theta0,options)
% Syntax:   [b,f,cov] = maxlike(fhandle,theta0,opt.alg,opt.cov,opt.out, GRAD)
% options: Optional argument, input as a struct or string array
%
% INPUT
%           fhandle:     the name of a function that returns
%                        a vector of log-likelihoods for all observations (first output)
%                        a vector of scores for all observations (second output - optional)
%                        fct must have a single input: the parameter vector
%           For example: [ll,s]=llfun(theta)
%
%           theta0:      starting values (Kx1 vector)
%
%           options.alg:         string, indicator for optimization method:
%                        = 'NR'     Newton-Raphson
%                        = 'BHHH'   Berndt, Hall, Hall, Hausman (Default)
%                                   (Uses outer porduct of scores to approximate hessian) 
%
%           options.cov:         scalar, type of covariance matrix of parameters,
%                        = 'A',   inverse of the Hessian, inv(A)/N (default)
%                        = 'B',   inverse of the outer-product of the gradient, inv(B)/N
%                        = 'R',   Robust covariance matrix, inv(A)*B*inv(A)/N
% 
%           options.out:         scalar, indicator for output:
%                        = 0, no output 
%                        = 1, print summary output at FINAL iteration (default)
%                        = 2, print summary output at EACH iteration
% 
%           options.grad:    scalar, indicator analytical gradient:
%                        = 1, use user-supplied numerical gradient (default=0)
%
% OUTPUT
%           b:           Kx1 vector, Maximum Likelihood estimates
%           f:           scalar, function at minimum/maximum (mean log-likelihood)
%           cov:         KxK matrix, covariance matrix of coefficients
%
% NOTE      Some times likelifunction have more than one input, e.g. [ll,s]=llfun(w, theta, model)
%           in this case you can still use maximize by passing an in-line function to maximize. 
%           For example:
%           [thetahat,f,cov]=maximize_g(@(theta) llfun(w,theta, 'llprobit'),theta, options)


% ------------Set default options----------------
opt = struct( ...
    'alg','BHHH', ...
    'cov','A', ...
    'out',1, ...
    'grad',0, ...
    'tol',1e-5, ...
    'maxhalf',20, ...
    'stepup','FALSE', ...
    'maxiter',100 ...
    );

% -----------------------------------------------
if nargin > 2
    % ------------Check user-supplied inputs----------------
    if isstruct(options)+iscell(options)~=1
        error('Optional inputs must be specified as a struct or cell array')
    end
    
    % ------------Replace with user-specified inputs----------------
    % Convert to struct, if input is cell array
    if iscell(options)
        for i=1:1:length(options)/2
            ArrayVar(i) = options(i*2-1);     % array with names
            ArrayVal(i) = options(i*2);       % array with values
        end
        options=cell2struct(ArrayVal', ArrayVar',1); % Transform the arrays to a struct
    end
    % Replace default values with user-specified values
    is = isfield(options,fieldnames(opt));    % =1 if input specified by user
    s = fieldnames(opt);
    for arg=1:length(s)
        if is(arg)==1
            opt.(char(s(arg)))=options.(char(s(arg)));
        end
    end
end
% Done specifying options

tic;                            % Start stopwatch timer

% Initialize variables
it=0;                           % Iteration counter
lambda=1;                       % Stepsize
theta1=theta0;

if opt.grad==1;
    [f0,s]=fhandle(theta0);     % Nx1 vectors of likelihhod contributions and analytical scores
    f0=mean(f0);                 % Likelihood function at theta0
else
    f0=mean(fhandle(theta0),1);  % Likelihood function at theta0
    s=gradp(fhandle,theta0);    % Nummerical score
end;

N=size(s,1);                    % Number of observations
g=mean(s,1)';                   % Gradient
if strcmp(opt.alg,'NR')         % Newton-Raphson: Use Hessian
    h=hessp(fhandle,theta0);    % Average of hessians over the cross-section
    m=(g'/(s'*s/N))*g;          % Convergence measure (equivalent to g'*inv(-h)*g - but faster)
elseif strcmp(opt.alg,'BHHH')
    h=-s'*s/N;
    m=(g'/(-h))*g;              % Convergence measure (equivalent to g'*inv(-h)*g - but faster)
end;


% MAIN LOOP
while abs(m)>opt.tol             % Loop until convergence
    theta0=theta1;               % set initial value of theta
    d=lambda*((-h)\g);           % Direction from theta0 - equivalent to d=lambda*inv(-h)*g
    f=mean(fhandle(theta0+d),1); % Likelihood function at theta1=theta0+d
    if f>=f0;                    % If step result in an increase
        % One step-increase if possible
        if strcmp(opt.stepup,'TRUE')       
            lambda_=lambda*2;
            d_=lambda_*((-h)\g);           % Direction from theta0 - equivalent to d=lambda*inv(-h)*g
            f_=mean(fhandle(theta0+d_),1); % Likelihood function at theta1=theta0+d
            if f_>=f
                lambda=lambda_;
                d=d_;
                f=f_;
            end
        end
        it=it+1;
        theta1=theta0+d;        % Update parameter value (see 12.81 wooldridge)
        f0=f;
        
        % PRINT ITERATION INFO
        if opt.out>=2
            format short                                                           % Set output format for numeric variables in Command Window
            disp(' ');
            disp(['Iteration:            ' num2str(it)]);
            disp(['Function value:       ' num2str(f)]);
            disp(['Convergence measure:  ' num2str(m)]);
            disp(['Stepsize:             ' num2str(lambda)]);
            disp('=====================================================================');
            disp('    Param.    Param. value   Direction   Gradient ');
            disp([(1:1:size(theta1,1))', theta1, d, g]);
            disp('---------------------------------------------------------------------');
            disp(' ');

            pause(0);
        end;
        
        lambda=1;       % Reset lambda
        
        if opt.grad==1;     % Use analytical gradient
            [f,s]=fhandle(theta1);
            f=mean(f);
            if strcmp(opt.alg,'NR');       % NEWTON-RAPHSON %
                h=gradp(@(theta) sumg(fhandle,theta),theta1)/N;
            elseif strcmp(opt.alg,'BHHH');     %--- BHHH ---%
                h=-s'*s/N;  % Use outer product of the gradient as approximation to hessian
            end
        else            % Use numerical score
            s=gradp(fhandle,theta1);
            if strcmp(opt.alg,'NR')       % NEWTON-RAPHSON %
                h=hessp(fhandle,theta1);
            else            %--- BHHH ---%
                h=-s'*s/N;  % Use outer product of the gradient as approximation to hessian
            end
        end
        
        g=mean(s,1)';       % Gradient
        m=g'/(s'*s/N)*g;    % Convergence measure
    else                    % If step result in an increase
        lambda=lambda/2;    % Half the stepsize
        if opt.out>=2;
            disp('Linesearching....');
        end;
        if lambda<= 2^-opt.maxhalf;
            ret=1;
            break
        end;
    end;
    if it >= opt.maxiter
        ret=2; 
        break
    end;
    
    ret=0;
end;

% CALCULATE STANDARD ERRORS
cov=zeros(size(theta0,1));                   % Initialize covariance matrix

if ((opt.cov=='A')+(opt.cov=='R'))  % Use inverse of hessian as covariance estimator
    if strcmp(opt.alg,'NR')~=1;
        if opt.grad==1;     % Use analytical score
            % to compute hessian as numerical gradient of analytical gradient
            h=gradp(@(theta) sumg(fhandle,theta),theta1)/N;
        else            % Use pure nummerical hessian
            h=hessp(fhandle,theta1);
        end
    end
end

if strcmp(opt.cov,'A');                % Use negaive of inverse hessian as covariance estimator                       
    cov=inv(-h)/N;
elseif strcmp(opt.cov,'B');            % Use inverse of outer-product of the score as covariance estimator
    cov=inv(s'*s/N)/N;
elseif strcmp(opt.cov,'R');
    cov=(-h)\(s'*s/N)/(-h)/N;     % Robust covariance matrix, eq to cov=inv(-h)*(s'*s/N)*inv(-h)/N
end

se=sqrt(diag(cov));  % Obtain standard errors

if opt.out>=1  % PRINT FINAL opt.outPUT
     if (ret==0);
        disp(' ');disp(' ');disp(' ');
        disp('                 ****************************');
        disp('                 **  Convergence achieved  **');
        disp('                 ****************************');
        disp('');
    elseif ret==1;
        disp('************************************************************************************');
        disp('** WARNING - NO CONVERGENCE: Stepsize too small - Still no increase in likelihood **');
        disp('************************************************************************************');
    elseif ret==2;
        disp('*******************************************************************************');
        disp('** WARNING - NO CONVERGENCE: Maximum iterations reached without convergence! **');
        disp('*******************************************************************************');
    end
    disp('Maximization algorithm: ');
    disp(opt.alg);
    telapsed=toc;
    format short
    disp(['Total number of iterations  ' num2str(it)]);
    disp(['CPU time (seconds):         ' num2str(telapsed)]);
    disp(['Mean Log-likelihood:        ' num2str(f)]);
    disp(['Number of observations      ' num2str(N)]);
    disp(['Convergence measure:        ' num2str(m)]);
    disp('=====================================================================');
    disp('   Param  Param value  Gradient    s.e.');
    format shortg
    disp([(1:1:size(theta1,1))', theta1, g, se]);
    disp('---------------------------------------------------------------------');
    if opt.cov>=1
        disp('NOTE: Covariance matrix of the parameters computed by the following method: ');
    end;
    if strcmp(opt.cov,'A'); disp('Inverse of computed Hessian'); end;
    if strcmp(opt.cov,'B'); disp('Inverse of outer product of the gradient'); end;
    if strcmp(opt.cov,'R'); disp('Robust'); end;
    disp(' ');
    disp(' ');
end
end

function g=sumg(fhandle,theta) % Nummerical computed hessian baed on gradient
[f,s]=fhandle(theta);
g=sum(s)';
end

%  Purpose:    Computes the gradient vector or matrix (Jacobian) of a
%              vector-valued function that has been defined in a procedure.
%              Single-sided (forward difference) gradients are computed.
%
%  Format:     g = gradp(@fhandle,x0);
%
%  Input:      @fhandle    scalar, m-file function pointer to a vector-valued function:
%
%                                          f:Kx1 -> Nx1
%
%              x0           Kx1 vector of points at which to compute gradient.
%
%  Output:     g            NxK matrix containing the gradients of f with respect
%                           to the variable x at x0.

function g = gradp(fhandle,x0)

% CHECK FOR COMPLEX INPUT
realcheck = isreal(x0); 
if realcheck == 0
    error('ERROR: Not implemented for complex matrices.');
else
  x0 = real(x0);
end;

f0 = fhandle(x0);
n = size(f0,1);
k = size(x0,1);
grdd = zeros(n,k);

% COMPUTATION OF STEPSIZE (dh)
    ax0 = abs(x0);
    if x0~=0
        dax0 = x0./ax0;
    else
        dax0 = 1;
    end;
    max0= max([ax0,(1e-2)*ones(size(x0,1),1)],[],2); 
    dh = (1e-8)*max0.*dax0;
 
    xdh = x0+dh;
    dh = xdh-x0;     
    
    reshapex0=zeros(k,k);
    for j=1:k
        reshapex0(:,j) = x0(:,1);
    end;
    
    arg=reshapex0;
    for j=1:k
        arg(j,j) = xdh(j,1);
    end;
    
    for i=1:k
        grdd(:,i) = fhandle(arg(:,i));
    end;
    
    f02mat = (ones(k,1)*f0')';
    dh2mat = (ones(n,1)*dh');

    g = (grdd-f02mat)./(dh2mat);
    

end

%  Purpose:  Computes the matrix of the mean of second partial derivatives
%            (Hessian matrix) of a function defined by a procedure.
%
%  Format:   h = hessp(@fhandle,x0);
%
%  Inputs:   @fhandle     pointer to a single-valued function f(x), defined
%                         as a procedure, taking a single Kx1 vector
%                         argument (f:Kx1 -> 1x1).  
%
%            x0           Kx1 vector specifying the point at which the Hessian
%                         of f(x) is to be computed.
%
%  Output:   h            KxK matrix of means of second derivatives of f with respect
%                         to x at x0. This matrix will be symmetric.
%

function h = hessp(fhandle,x0)

% CHECK FOR COMPLEX INPUT 
realcheck = isreal(x0); 
if realcheck == 0
    error('ERROR: Not implemented for complex matrices.');
else
  x0 = real(x0);
end;

% INITIALIZATIONS
    k = size(x0,1);
    hessian = zeros(k,k);
    grdd = zeros(k,1);
    eps = 6.0554544523933429e-6;
    
    
% COMPUTATION OF STEPSIZE (dh) 
ax0 = abs(x0);
    if x0 ~= 0
        dax0 = x0./ax0;
    else
        dax0 = 1;
    end;
    max0= max([ax0,(1e-2)*ones(size(x0,1),1)],[],2);  
    dh = eps*max0.*dax0;
 
    xdh = x0+dh;
    dh = xdh-x0;
    dh2mat=(ones(k,1)*dh')';
    ee = eye(k).*dh2mat;
    
% COMPUTATION OF f0=f(x0) 
    f0=fhandle(x0);
    N=size(f0,1);
    f0 = sum(f0,1);

% COMPUTE FORWARD STEP 
    for i=1:k

        grdd(i,1) = sum(fhandle(x0+ee(:,i)),1);

    end;

% COMPUTE "DOUBLE" FORWARD STEP   
     for i=1:k
        for j=1:k

            hessian(i,j) = sum(fhandle(x0+(ee(:,i)+ee(:,j))),1);
            if i ~= j
                hessian(j,i) = hessian(i,j);
            end;

        end;
    end;

grdd2mat=(ones(k,1)*grdd')';
h = (((hessian - grdd2mat) - grdd2mat')+ f0)./ (dh2mat.*dh2mat');
h=h/N;
 
end


