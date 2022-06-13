

%% Example 2: decompose the whole night EEG
% In this example, we demonstrate decomposing the whole-night EEG by 
% parallel EMD, and use the criticle points to calculate instantaneous 
% frequency (defined by zero-crossing) 
% The sample signal is downloaded from Haaglanden Medisch Centrum sleep
% staging database of Physionet
% Alvarez-Estevez, D., & Rijsman, R. (2021). Haaglanden Medisch Centrum 
% sleep staging database (version 1.0.1). PhysioNet. https://doi.org/10.13026/7egw-0p30.
% https://www.physionet.org/content/hmc-sleep-staging/1.0.1/
% The original PSG is in EDF file. We converted it to .mat format

%% Load the data
clear; clc; close all;
addpath('../cuda')
load('../data/SN002_EEG.mat'); % EEG, SampleRate, and Channel are loaded
fs = SampleRate;
sleepscoring = readtable('SN002_sleepscoring.txt');

%% Reshape the EEG to matrix, each column is a epoch
epochL = 30*fs;
EEGA = buffer(EEG, epochL);

%% Decompose by parallel EMD
nm = floor(log2(epochL/2));
nsift = 10;

[out0,val0,idx0,len0,up0] = mex_gpuEMD_env(EEGA,nm, nsift);

% Reshpe the output
% %the IMFs should be matrices with the same shape as A
x_len = epochL; y_len = size(EEGA,2); 
out = reshape(out0, x_len, y_len, nm);
val = reshape(val0, x_len*2, y_len, nm);
idx = reshape(idx0, x_len*2, y_len, nm);
len = reshape(len0, y_len, nm);
up = reshape(up0, x_len, y_len, nm);

%% Analyze the frequency by criticle points (generalized zero-crossing)
% The generalized zero-crossing is propoased by Huang 2006
% For details, please see: https://ntrs.nasa.gov/citations/20080008712
timePowerPlot = [];
for iwin = 11:y_len
    % for each epoch
    
    criticalPoints = squeeze(idx(:,iwin,:));
    upperEnv = squeeze(up(:,iwin,:));
    nCritpts = len(iwin,:);
    % trimIMF with nCrit < 6;
    rmIMF = nCritpts < 30;
    criticalPoints(:,rmIMF) = [];
    upperEnv(:,rmIMF) = [];
    % calculate insFreq
    InsFreq1 = instFreq3(criticalPoints, epochL, fs, len(iwin,:));
    [fscale,sPowerPlot,~] = freqPowerPlot3(InsFreq1, upperEnv);
    timePowerPlot = [timePowerPlot,sPowerPlot'];
end

%% Parse sleep stage


sleepscoring.stageNum = nan(height(sleepscoring),1);
idxx = find(strcmp(sleepscoring.Annotation,'Sleep stage W')); sleepscoring.stageNum(idxx) = 0;
idxx = find(strcmp(sleepscoring.Annotation,'Sleep stage N1')); sleepscoring.stageNum(idxx) = 1;
idxx = find(strcmp(sleepscoring.Annotation,'Sleep stage N2')); sleepscoring.stageNum(idxx) = 2;
idxx = find(strcmp(sleepscoring.Annotation,'Sleep stage N3')); sleepscoring.stageNum(idxx) = 3;
idxx = find(strcmp(sleepscoring.Annotation,'Sleep stage R')); sleepscoring.stageNum(idxx) = 5;
idxx = isnan(sleepscoring.stageNum); sleepscoring(idxx,:) = [];
%%
Nepoch = min(height(sleepscoring), size(timePowerPlot,2));

time_hr = 30*(1:Nepoch)/60/60;
figure;
subplot(3,1,1); plot(time_hr, sleepscoring.stageNum(1:Nepoch));
xlim([0,time_hr(end)])
subplot(3,1,2:3); imagesc(time_hr,fscale,timePowerPlot(10:end,1:Nepoch).^0.25);
axis xy
ylim([0, 30])

%colorbar()
%figure; plot(timePowerPlot(10:end,:))

