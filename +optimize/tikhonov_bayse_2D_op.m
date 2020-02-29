
% tikhonov_bayse_2D_op  Finds optimal lambda and alpha for Tikhonov solver 
% using known distribution, x
% Author: Arash Naseri, Timothy Sipkens, 2020-02-28
%-------------------------------------------------------------------------%
% Inputs:
%   A       Model matrix
%   b       Data
%   C       Model matrix 2
%   d       Data set 2
%   n       Length of first dimension of solution
%   lambda  Regularization parameter
%   span    Range for 1/Sf, two entry vector
%   x_ex    Exact distribution project to current basis
%   order   Order of regularization     (Optional, default is 1)
%   xi      Initial guess for solver    (Optional, default is zeros)
%   solver  Solver                      (Optional, default is interior-point)
%
% Outputs:
%   x       Regularized estimate
%   Lx      Tikhonov matrix
%=========================================================================%

function [x,out,chi] = tikhonov_bayse_2D_op(A,b,C,d,span1,span2,order,n,x_ex,xi,solver)

%-- Parse inputs ---------------------------------------------------------%
if ~exist('order','var'); order = []; end
if ~exist('xi','var'); xi = []; end
if ~exist('x_ex','var'); x_ex = []; end
if ~exist('solver','var'); solver = []; end
%-------------------------------------------------------------------------%
%-- Compute credence, fit, and Bayes factor --%
% Initially meshing the domain of (lambda, alpha ) to roughly find the 
% location of global extremum of B
lambda = logspace(log10(span1(1)),log10(span1(1)),10);
alpha =  logspace(log10(span2(1)),log10(span2(1)),10);
[X,Y]=meshgrid(lambda,alpha);
Param=[X(:),Y(:)];
x_length = size(A,2);

Lpr0 = invert.tikhonov_lpr(order,n,x_length); % get Tikhonov matrix
tools.textbar(0);

for ii=1:length(Param)
    out(ii).lambda = Param(ii,1); % store regularization parameter
    out(ii).alpha = Param(ii,2); % store regularization parameter
    %-- Perform inversion --%
    [out(ii).x,~,Lpr0] = invert.tikhonov(...
        [Param(ii,2).*A;C],[Param(ii,2).*b;d],Param(ii,1),Lpr0,[],xi,solver);
    %-- Store ||Ax-b|| and Euclidean error --%
    if ~isempty(x_ex); out(ii).chi = norm(out(ii).x-x_ex); end
     out(ii).Axb = norm(A*out(ii).x-b);
    
    %-- Compute credence, fit, and Bayes factor --%
    [out(ii).B,out(ii).F,out(ii).C] = ...
        optimize.bayesf_b([Param(ii,2)*A;C],[Param(ii,2)*b;d],invert.tikhonov...
        ([Param(ii,2)*A;C],[Param(ii,2)*b;d],Param(ii,1),Lpr0,[],xi,solver),Lpr0,Param(ii,1));
    tools.textbar((length(Param)-ii+1)/length(Param));
end


if ~isempty(x_ex) % if exact solution is supplied
    [~,ind_min] = min([out.chi]);
else
    ind_min = [];
end
x = out(ind_min).x;
out(1).ind_min = ind_min;

disp('Optimizing Tikhonov regularization:');

    %-- Perform inversion --%
 
fun=@(lambda) log(-1*optimize.bayesf_b([lambda(2)*A;C],[lambda(2)*b;d],invert.tikhonov...
    ([lambda(2)*A;C],[lambda(2)*b;d],lambda(1),Lpr0,[],xi,solver),Lpr0,lambda(1)));

lambda_guess = [out(ind_min).lambda out(ind_min).alpha]; % initial guess for fminsearch

options = optimset('TolFun',10^-8,'TolX',10^-8,'Display','iter');

lambdaOpt = fminsearch(fun, lambda_guess,options);

lambda=lambdaOpt(1);
alpha=lambdaOpt(2);

x=invert.tikhonov([alpha*A;C],[alpha*b;d],lambda,Lpr0,[],xi,solver);
chi=100*norm(x-x_ex)/norm(x_ex);

% % -- Compute credence, fit, and Bayes factor --%
% % out.Lpr = Lpr0; % store Lpr structure
% % to save memory, only output Lpr structure
% % Lpr for any lambda can be found using scalar multiplication
% % Gpo_inv = A'*A+Lpr'*Lpr; <- can be done is post-process

end

