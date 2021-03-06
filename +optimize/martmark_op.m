
% MARTMARK_OP  Finds optimal smoothing for MART with Markowski-type smoothing.
% Author: Arash Naseri, Timothy Sipkens, 2030-02-06
%-------------------------------------------------------------------------%
% Inputs:
%   A           Model matrix
%   b           Data
%   Lb          Cholesky factorization of inverse covariance matrix
%   n           Length of first dimension of solution, used in smoothing
%   xi          Initial guess
%   iter        Max. number of iterations of Twomey_Markowski algorithm
%   span        Range for 1/Sf, two entry vector
%   x_ex        Exact distribution project to current basis
%   opt_smooth  Type of smoothing to apply      (Optional, default is 'Buckley')
%
% Outputs:
%   x           Estimate
%   Sf          Estimate of optimized value of Sf
%   out         Struct containing detailed information about optimization
%=========================================================================%

function [x,Sf,out] = martmark_op(A,b,Lb,n_grid,xi,iter,span,x_ex,opt_smooth)


%-- Parse inputs ---------------------------------------------------------%
if ~exist('opt_smooth','var'); opt_smooth = []; end
if isempty(opt_smooth); opt_smooth = 'Buckley'; end

if ~exist('x_ex','var'); x_ex = []; end
%-------------------------------------------------------------------------%


x_fun = @(Sf) invert.martmark(A,b,Lb,n_grid,xi,iter,opt_smooth,Sf);

Sf = logspace(log10(span(1)),log10(span(2)),35);

disp(' ');
disp('Optimizing MART-Markowski smoothing:');
tools.textbar(0);
disp(' ');
for ii=length(Sf):-1:1 % loop through values of Sf
    out(ii).Sf = Sf(ii);
    out(ii).x = x_fun(out(ii).Sf);
    if ~isempty(x_ex); out(ii).chi = norm(out(ii).x-x_ex); end
    out(ii).Axb = norm(A*out(ii).x-b);
    
    disp(' ');
    disp('Optimizing MART-Markowski smoothing:');
    tools.textbar(0); % reinitialize textbar
    tools.textbar((length(Sf)-ii+1)/length(Sf));
    disp(' ');
end

if ~isempty(x_ex)
    [~,ind_min] = min([out.chi]);
else
    ind_min = [];
end
Sf = out(ind_min).Sf;
x = out(ind_min).x;

if or(ind_min==1,ind_min==length(out))
    disp('Minimum occured at the edge of specified span for Sf.');
end

end


