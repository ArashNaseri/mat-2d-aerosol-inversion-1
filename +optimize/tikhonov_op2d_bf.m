
% TIKHONOV_OP2D_BF  Finds optimal lambda and alhpa for Tikhonov solver.
% Author: Arash Naseri, Timothy Sipkens, 2020-02-28
%=========================================================================%

function [x,lambda,alpha,out,chi] = tikhonov_op2d_bf(A,b,C,d,span1,span2,order,n,x_ex,xi,solver)


%-- Parse inputs ---------------------------------------------------------%
if ~exist('order','var'); order = []; end
if ~exist('xi','var'); xi = []; end
if ~exist('x_ex','var'); x_ex = []; end
if ~exist('solver','var'); solver = []; end
%-------------------------------------------------------------------------%


%-- Compute credence, fit, and Bayes factor ------------------------------%
% Initially meshing the domain of (lambda, alpha ) to roughly find the 
% location of global extremum of B
lambda = logspace(log10(span1(1)),log10(span1(2)),4);
alpha =  logspace(log10(span2(1)),log10(span2(2)),4);
[lambda_mat,alpha_mat] = meshgrid(lambda,alpha);
param = [lambda_mat(:),alpha_mat(:)]; % set of lambda and alpha to consider
x_length = size(A,2);

Lpr0 = invert.tikhonov_lpr(order,n,x_length); % get Tikhonov matrix

tools.textbar(0);
for ii=length(param):-1:1 % reverse loop to pre-allocate
    out(ii).lambda = param(ii,1); % store regularization parameter
    out(ii).alpha = param(ii,2); % store regularization parameter
    
    %-- Perform inversion --%
    [out(ii).x,~,Lpr0] = invert.tikhonov(...
        [A;C],[param(ii,2)*b;d],param(ii,1),Lpr0,[],xi,solver);
    %-- Store ||Ax-b|| and Euclidean error --%
    if ~isempty(x_ex); out(ii).chi = norm(out(ii).x-x_ex); end
     out(ii).Axb = norm(A*out(ii).x-b);
    
    %-- Compute credence, fit, and Bayes factor --%
    out(ii).x = invert.tikhonov([A;C],[param(ii,2)*b;d],...
        param(ii,1),Lpr0,[],xi,solver);
    [out(ii).B,out(ii).F,out(ii).C] = ...
        optimize.bayesf_b([A;C],[param(ii,2)*b;d],...
        out(ii).x,Lpr0,param(ii,1));
    tools.textbar((length(param)-ii+1)/length(param));
end

%-- Record a rough estimate of the solution --%
[~,ind_min] = max([out.B]); % get optimal w.r.t. Bayes factor
out(1).ind_min = ind_min;
%-------------------------------------------------------------------------%


%-- Add fminsearch step to optimize parameter set --------------------%
disp('Optimizing Tikhonov regularization:');
fun = @(lambda) -optimize.bayesf_b([A;C],[lambda(2)*b;d],invert.tikhonov...
    ([A;C],[lambda(2)*b;d],lambda(1),Lpr0,[],xi,solver),Lpr0,lambda(1));

y0 = [out(ind_min).lambda out(ind_min).alpha]; % initial guess for fminsearch
options = optimset('MaxFunEvals',300,'TolFun',10^-4,'TolX',10^-4,'PlotFcns',@optimplotfval,'Display','iter');
y1 = fminsearch(fun,y0,options); %get optimal lambda and alpha

lambda = y1(1); % assign output variables
alpha = y1(2);
disp('Complete.');
%---------------------------------------------------------------------%


x = invert.tikhonov(...
    [A;C],[alpha*b;d],lambda,Lpr0,[],xi,solver);
if and(exist('x_ex','var'),norm(x_ex)~=0);chi = norm(x-x_ex);else chi = []; end

end
