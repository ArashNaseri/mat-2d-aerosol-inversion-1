% This is the main MATLAB file for running 2D inversions. Please make sure
% to look through the README.txt file for specific instructions on running
% this code

% We acknowledge the support of Kanomax LLC and Kanomax-FMT for the development
% of this code.

clear

% format long

%% User inputs for error

Qae = 1.02E-3/60; % aerosol flowrate [m^3/s]
t_ms = 5; % measurement time at each channel [s]
errorCPC = 0.10; % standard error the CPC (TSI 3776: 10%, TSI 3786: 12%)
resultsFile = 'resultsFile';

%% Defining constants/environmental factors

mu = 1.82E-5; % gas viscosity [Pa*s]
e = 1.6022E-19; % electron charge [C]
T = 298; % gas temperature [T]

%% Read in measured data 
% This section reads in 'InputData.csv'

% response = csvread('+buckley_program.csv',0,2);      % response is the variable describing 
                                              % the measurements at channel i,j

load('+buckley_program/response_Hogan.mat');
                                              
[dpStarLen,mpStarLen] = size(response);

% dpStar = csvread('+Old/InputFile.csv',0,0,[0,0,dpStarLen-1,0]);
load('+buckley_program/mpdp_vec_Hogan.mat');
dpStar = dpStar(dpStar ~= 0);
dpStarLen = length(dpStar);                   % dpStar is an array of the mobility
                                              % diameters at which DMA
                                              % measurements were performed

% mpStar = csvread('+Old/InputFile.csv',0,1,[0,1,mpStarLen-1,1]);

mpStar = mpStar(mpStar ~= 0);
mpStarLen = length(mpStar);                   % mpStar is an array of the masses
                                              % defined by the APM measurements

response = response(1:dpStarLen,1:mpStarLen);
response = response.*(response > 0.9);		  % this line sets responses less than 0.1 #/cc to 0. Comment 
											  % this out if you'd like to disclude it.

errorCounting = (Qae*t_ms*response*1E6).^0.5/(Qae*1E6*t_ms); % response is multiplied by 1000 here since
                                              % it's given in cm^(-3)
                                              % rather than m^(-3)
                                              
% errorCounting = 0;                            % Comment this out for inverting real measurements. Counting error is 0 for inversion of simulated distributions.

% The errorSource term is meant to represent error due to source variation. To
% get an estimate of this term, perform several APM scans of DMA-selected
% particles which represent the full range of measured concentrations in
% your scan; i.e. perform several APM scans at the DMA voltage at which the
% largest range of concentrations was measured. Calculate the standard
% deviation of concentration at each APM setting, and fit the normalized
% standard deviation (normed by mean concentration) as a function of mean
% concentration. Typically, this value will be constant for most measured
% concentrations > 10 #/cc. For less than 10 #/cc, counting statistics
% become important and increase the overall error variation.
% For the data included with this file, for
% instance, a standard deviation of 11.9% was found, and so the errorSource
% term is 11.9% of the response. 

errorSource = 0.119.*response; 
errorBuffer = 10; % this term is included to reduce the overall effect of
                    % small concentration measurements on the chi-squared
                    % error; inclusion of this term biases the inversion
                    % towards using the higher concentration measurements
                    % as the representation of the overall 2D size-mass
                    % distribution.

error = ((errorCPC.*response).^2 + errorCounting.^2 + errorSource.^2 + errorBuffer^2).^(1/2);   % overall error calculation

%% Initial Guess & Transfer Function Calculation
% calc centroid mobility for DMA transmission

invZpStar = 3*pi*mu.*(dpStar)./(e*buckley_program.Cc(dpStar));

ZpStarLen = dpStarLen;

[ll,mm,~,~,~,~,~] = buckley_program.InversionParameters;

% initialize necessary arrays for inversion/transfer function description

dn_ddpdmp = zeros(ZpStarLen,mpStarLen);
DMATfer1 = zeros(ll,ZpStarLen); 
DMATfer2 = zeros(ll,ZpStarLen); 
APMTfer1 = zeros(mm,mpStarLen);
APMTfer2 = zeros(mm,mpStarLen);

GDMA1 = zeros(ll,ZpStarLen);
GDMA2 = zeros(ll,ZpStarLen);

DMAInt1 = ones(1,ZpStarLen);
DMAInt2 = ones(2,ZpStarLen);


% calc array of diameters for distribution description

dpLow = min(dpStar); 
dpLow = 0.1*dpLow;
dpHigh = max(dpStar);
dpHigh = 1.1*dpHigh;
dp4Int = logspace(log10(dpLow),log10(dpHigh),ll);% calc actual array of dp

invZp4Int = 3*pi*mu.*(dp4Int )./(e*buckley_program.Cc(dp4Int ));  % array of mobilities

% calc array of masses for distribution description

mpLow = min(mpStar);
mpLow = 0.1*mpLow;
mpHigh = max(mpStar);
mpHigh = 1.1*mpHigh;
mp4Int = logspace(log10(mpLow),log10(mpHigh),mm);

%{
% Alternate setup --------------------------------------------
load('sample_dp_mp.mat');
mp4Int = mp.*1e-18;
dp4Int = dp.*1e-9;
mpStar = mp0'.*1e-18;
dpStar = dp0'.*1e-9;
invZp4Int = 3*pi*mu.*(dp4Int )./(e*buckley_program.Cc(dp4Int ));  % array of mobilities
mpStarLen = length(mpStar);
dpStarLen = length(dpStar);
invZpStar = 3*pi*mu.*(dpStar)./(e*buckley_program.Cc(dpStar));
ZpStarLen = dpStarLen;
ll = length(dp4Int);
mm = length(mp4Int);
dn_ddpdmp = zeros(ZpStarLen,mpStarLen);
DMATfer1 = zeros(ll,ZpStarLen); 
DMATfer2 = zeros(ll,ZpStarLen); 
APMTfer1 = zeros(mm,mpStarLen);
APMTfer2 = zeros(mm,mpStarLen);
GDMA1 = zeros(ll,ZpStarLen);
GDMA2 = zeros(ll,ZpStarLen);
DMAInt1 = ones(1,ZpStarLen);
DMAInt2 = ones(2,ZpStarLen);
%}

% Charged Fraction: calc charged fraction of particles for bipolar
% diffusion charging based on Gopalakrishnan (w/ subroutine f_charge) 
% or wiedensohler (w/ subroutine f_charge_W).

[~,f1,f2] = buckley_program.f_charge(dp4Int);
f1 = transpose(f1);
f2 = transpose(f2);

Gtot1 = cell(ZpStarLen,mpStarLen);
Gtot2 = cell(ZpStarLen,mpStarLen);


%% Calculate transfer functions and integrate for initial guess

for i = 1:ZpStarLen % for loop operates over each measurement channel, i. i.e. each DMA voltage
        
    DMATfer1(:,i) = buckley_program.DMATransfer(invZp4Int,invZpStar(i),dp4Int,1);
    DMATfer2(:,i) = buckley_program.DMATransfer(invZp4Int,invZpStar(i),dp4Int,2); 

    GDMA1(:,i) = f1.*DMATfer1(:,i); % calc Kernels
    GDMA2(:,i) = f2.*DMATfer2(:,i); % doubly charged


    for j = 1:mpStarLen

        APMTfer1(:,j) = buckley_program.APMTransfer(1,transpose(mp4Int), invZpStar(i), mpStar(j)); 
        APMTfer2(:,j) = buckley_program.APMTransfer(2,transpose(mp4Int), invZpStar(i), mpStar(j));
        
        Gtot1{i,j} = sparse(transpose(transpose(ones(ll,mm).*GDMA1(:,i)).*APMTfer1(:,j))); 
        Gtot2{i,j} = sparse(transpose(transpose(ones(ll,mm).*GDMA2(:,i)).*APMTfer2(:,j)));
        
        G1 = Gtot1{i,j};
        G2 = Gtot2{i,j};

        totalInt1(i,j) = trapz(mp4Int,trapz(dp4Int,G1,1)); 
        totalInt2(i,j) = trapz(mp4Int,trapz(dp4Int,G2,1));
        
        dn_ddpdmp(i,j) = response(i,j)/(totalInt1(i,j) + totalInt2(i,j));

    end

end

dn_ddpdmp = transpose(dn_ddpdmp);


%% Interpolation to make the distribution the same size as the integrating range 

% part of original code -----------------------
mp4Int = mp4Int';
dn_ddpdmp = interp2(dpStar,mpStar,dn_ddpdmp,dp4Int,mp4Int,'linear',1E22); 

dn_ddpdmp = sparse(transpose(dn_ddpdmp));
dn_ddpdmp(isnan(dn_ddpdmp))=0;

%{
% Alternate ------------
ntot = 1e6;
rho1 = 1e3;
dg1 = 20e-9;
sg1 = 1.15;
rho2 = 1e2;
dg2 = 100e-9;
sg2 = 1.4;
sm = 0.2;
Vm = @(dp,rho) rho.*pi.*(dp).^3./6;
p_norm = @(dp,mp,rho) 1./(sqrt(2*pi).*8e-22.*sm).*...
    exp(-((mp'*ones(1,length(dp))-ones(length(mp),1)*Vm(dp,rho))./...
    (sqrt(2)*(ones(length(mp),1)*Vm(dp,rho).*sm))).^2);
p1 = @(dp,mp) bsxfun(@times,lognpdf(dp,log(dg1),log(sg1)),...
    p_norm(dp,mp,rho1));
p2 = @(dp,mp) bsxfun(@times,lognpdf(dp,log(dg2),log(sg2)),...
    p_norm(dp,mp,rho2));
p = @(dp,mp) ntot.*(p1(dp,mp)+p2(dp,mp));
dn_ddpdmp = p(dp4Int,mp4Int)';
%}

%% Calc Initial Error

y_1 = zeros(length(response),1); 
y_2 = zeros(length(response),1);

% Calculate expected measurement (y) based on current distribution
for i = 1:ZpStarLen
         
    for j = 1:mpStarLen
        
        G1 = Gtot1{i,j}.*dn_ddpdmp;
        G2 = Gtot2{i,j}.*dn_ddpdmp;

        totalInt1(i,j) = trapz(mp4Int,trapz(dp4Int,G1,1)); % totalInt
        totalInt2(i,j) = trapz(mp4Int,trapz(dp4Int,G2,1));
        
    end
end

y = totalInt1 + totalInt2;

load('..\Program - LII\LII Program 3.9\+CENIDE\viridis.mat');
gca = figure(2);
colormap(gca,cm);
imagesc(log10(dp4Int),log10(mp4Int),dn_ddpdmp');

gca = figure(3);
colormap(gca,cm);
imagesc(log10(dp4Star),log10(mp4Star),y');

% calculate chi-squared (SIGMA) for all channels where measurement was > 0.01 #/cc
% (relevant data)

SIGMA_new = 0;
numErrorCalcs = 0;

for i  = 1:ZpStarLen
    for j = 1:mpStarLen
        
        if response(i,j) > 0.01
            SIGMA_new = SIGMA_new + ((response(i,j) - y(i,j))/error(i,j))^2;
            numErrorCalcs = numErrorCalcs + 1;
        end
    end    
end

SIGMA_new = SIGMA_new/numErrorCalcs;
SIGMA = SIGMA_new;

%% Perform the Twomey-Markowski Algorithm

% Smooth before entering Twomey

[dn_ddpdmp, SIGMA] = buckley_program.smooth(dn_ddpdmp, 5, response, Gtot1, Gtot2, dp4Int, mp4Int, SIGMA, error);


% Initial Twomey run:

LIMIT = 0.97; % LIMIT = 0.97 creates hysteresis between smooth and Twomey loops.
[dn_ddpdmp, R_dp, R_mp, SIGMA, y] = buckley_program.Twomey(dn_ddpdmp, Gtot1, Gtot2, dp4Int, mp4Int, response, LIMIT, error, SIGMA);

q = 1;
R_dpMat = R_dp; % initialize roughness arrays
R_mpMat = R_mp;

% smooth initial Twomey solution
[dn_ddpdmp, SIGMA] = buckley_program.smooth(dn_ddpdmp, 5, response, Gtot1, Gtot2, dp4Int, mp4Int, SIGMA, error);


%% Twomey-Markowski While Loop (Main Algorithm)

R_dp0 = R_dp*1.1;
R_mp0 = R_mp*1.1;

maxIter = 600;


while q < maxIter
    
    if 1.0*R_dp0 < R_dp && R_mp0 < R_mp
        dn_ddpdmp = dn_ddpdmp0;
        disp('roughness increased too much after twomey loop')
        break
    end
    dn_ddpdmp0 = dn_ddpdmp;
    R_dp0 = R_dp;
    R_mp0 = R_mp;
    [dn_ddpdmp, R_dp, R_mp, SIGMA, y] = buckley_program.Twomey(dn_ddpdmp, Gtot1, Gtot2, dp4Int, mp4Int, response, LIMIT, error, SIGMA);
    q = q + 1;
    R_mpMat = [R_mpMat, R_mp];
    R_dpMat(q) = R_dp;
    [dn_ddpdmp, SIGMA] = buckley_program.smooth(dn_ddpdmp, 25, response, Gtot1, Gtot2, dp4Int, mp4Int, SIGMA, error);

end


if q > maxIter
    display({'max iterations of ', num2str(q),' completed'})
end

display({'Final SIGMA = ', num2str(SIGMA(end))});
display({'q = ', num2str(q)});

%% calculating dn/ddp and dn/dmp

dn_ddp = trapz(mp4Int,dn_ddpdmp,2);
dM_ddp = trapz(mp4Int,dn_ddpdmp.*mp4Int',2);

%% write results

test = full(dn_ddpdmp);
csvwrite(['Results/',resultsFile,'dp.csv'],dp4Int)
csvwrite(['Results/',resultsFile,'mp.csv'],mp4Int)
csvwrite(['Results/',resultsFile,'distribution.csv'],test)
csvwrite(['Results/',resultsFile,'dn_ddp.csv'],dn_ddp)
csvwrite(['Results/',resultsFile,'dM_ddp.csv'],dM_ddp)
csvwrite(['Results/',resultsFile,'ChiSquared.csv'],SIGMA)