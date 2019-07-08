
% PHANTOM   Class containing properties and methods for storing phantoms.
% Author:   Timothy Sipkens, 2019-07-08
%=========================================================================%

%-- Class definition -----------------------------------------------------%
classdef Phantom
    
    %-- Phantom properties -----------------------------------------------%
    properties
        n_modes = []; % number of modes
        
        param@struct = struct(...
            'rho',[],... % effective density at mean diameter [kg/m3]
            'Dm',[],... % mass-mobility exponent
            'dg',[],... % mean mobility diameter [nm]
            'sg',[],... % standard deviation of mobility diameter
            'k',[],... % mass-mobility pre-exponential factor
            'sm',[],... % standard deviation of mass
            'opt_m','logn',... % form of probability distribution ('logn' or 'norm')
            'mg_fun',[]... % center of the mass distribution (function handle)
            );
        
        grid = []; % grid phantom is evaluated on (high dimension)
        x = []; % evaluated phantom
    end
    
    
    %-- Phantom methods --------------------------------------------------%
    methods
        %== PHANTOM ======================================================%
        %   Intialize phantom object.
        function obj = Phantom(name,span,param)
            
        	%-- Parse inputs ---------------------------------------------% 
            switch name
                case {'demonstration'}
                    obj.param(1).rho = 12000; % density of gold-ish
                    obj.param(1).Dm = 3;
                    obj.param(1).dg = 50;
                    obj.param(1).sg = 1.4;
                    obj.param(1).k = (obj.param(1).rho*pi/6).*...
                    	obj.param(1).dg^(3-obj.param(1).Dm);
                    obj.param(1).sm = 1.3;
                    obj.param(1).opt_m = 'logn';
                    
                    obj.param(2).rho = 500; % density of salt
                    obj.param(2).Dm = 2.3;
                    obj.param(2).dg = 200;
                    obj.param(2).sg = 1.4;
                    obj.param(2).k = (obj.param(2).rho*pi/6).*...
                    	obj.param(2).dg^(3-obj.param(2).Dm);
                    obj.param(2).sm = 1.3;
                    obj.param(2).opt_m = 'logn';
                    
                case {'soot-surrogate'}
                    
                    
                case {'Buckley','Hogan'}
                    
                    
                case {'narrow'}
                    
                    
                otherwise % for custom phantom
                    if ~exist('param','var'); error('Specify phantom.'); end
                    if isempty(param); error('Specify phantom.'); end
                    
                    for ii=1:length(param.fields)
                        
                    end
            end
            obj.n_modes = length(obj.param);
            
            %-- Evaluate phantom -----------------------------------------%
            [obj.x,obj.grid,obj.param.mg] = obj.gen_phantom(span);

        end
        %=================================================================%
            
        
        %== GEN_PHANTOM ==================================================%
        %   Generates a mass-mobiltiy distribution phantom.
        %   Author:     Timothy Sipkens, 2018-12-04
        function [x,grid,mg] = gen_phantom(obj,span)
            
            n_t = [540,550]; % resolution of phantom distribution
            grid = Grid(span,... 
                n_t,'logarithmic'); % generate grid of which to represent phantom
            
            %-- Evaluate phantom mass-mobility distribution ---------------%
            m0 = grid_t.elements(:,1);
            d0 = grid_t.elements(:,2);
            
            rho = @(d,k,Dm) 6*k./(pi*d.^(3-Dm));
            
            mg = @(d0,ll) log(1e-9.*rho(d0,obj.param(ll).k,obj.paramparam(ll).Dm).*...
                pi/6.*(d0.^3)); % geometric mean mass in fg
            
            x = zeros(length(m0),1);
            for ll=1:obj.n_modes % loop over different modes
                if strcmp(obj.param(ll).opt_m,'norm')
                    p_m = normpdf(m0,...
                        exp(mg(d0,ll)),obj.param(ll).sm.*exp(mg(d0,ll)));
                else
                    p_m = lognpdf(m0,mg(d0,ll),log(obj.param(ll).sm));
                end
                
                p_temp = lognpdf(d0,log(obj.param(ll).dg),log(obj.param(ll).sg)).*...
                    p_m;
                x = x+p_temp;
            end
            
            x = x./obj.n_modes;
            x = x.*(d0.*m0); % convert to [lnm,lnd]T space
        end
        %=================================================================%
        
        
        %-- Data visualization -------------------------------------------%
        [h,x] = plot_phantom(obj,x); % plots x on the grid
        %-----------------------------------------------------------------%
        
    end
    
end

