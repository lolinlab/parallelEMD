%Author: Daniel Abadjiev
%Date: July 21, 2020
% Modified by Hui-Wen, Feb 10, 2022.
%Description: This program calculates a vector which gives the power at
%each frequency, which comes from the amplitude of the IMFs at that
%frequency
%Based on nspplotf.m, but 1-D 
%Update October 13, 2020 to use windowStruct from holospec2 instead of the
%windowCol cell array from holospec

%function [fscale,sPowerPlot,allPowerPlot] = freqPowerPlot2(winStruct,fw0,fw1,ntp0,ntp1)
function [fscale,sPowerPlot,allPowerPlot] = freqPowerPlot3(insFreq1, insAmp1);
%Input -
%   insFreq1: instantaneous frequency of each IMF, row-wise
%   insAmp1: instantaneous amplitude of each IMF, row-wise
%   fw0         - the lowest frequency to look for
%   fw1         - the highest freuqency to look for
%   ntp0        - the start time (point)
%   ntp1        - the end time (point)
%Output - 
%   fscale      - same fscale as used in holospectrum
%   sPowerPlot  - sum of all power plots for each imf
%   allPowerPlot- each power plot, organized by imfNumber
if (nargin <2)
    warning('not enough inputs to freqPowerPlot.m, cancelling')
    return;
end

%% Initialize values
% insFreq1 = winStruct(1).insFreq1;
% insAmp1 = winStruct(1).insAmp1;

fw0 = 0.03;
fw1 = 40;
fw = fw1 - fw0;
ntp0 = 1;
ntp1 = size(insFreq1,2);


fres = 480; %The resolution, same as holospectrum resolution
fscale = linspace(fw0, fw1, fres);
imfNum = size(insFreq1,1);

%% Generate the powerplot
sPowerPlot = zeros(1,fres);
allPowerPlot = zeros(imfNum,fres);

for (i_imf = 1:imfNum)
    powerPlot = zeros(1,fres);
    numFreqPnts = zeros(1,fres);
    freqs = insFreq1(i_imf,:);
    amps = insAmp1(i_imf,:);
    %Not sure if necessary, I'm not squaring amps so maybe?
    amps = abs(amps);
    
    pf = round((fres-1)*(freqs-fw0)/fw)+1; %% why??
    
    %Do summing for powerPlot
    for x=ntp0:ntp1
        freqidx = pf(x);
        
        if (freqidx >= 1 && freqidx <= fres) 
            powerPlot(freqidx)=powerPlot(freqidx)+amps(x);
            numFreqPnts(freqidx) = numFreqPnts(freqidx) + 1; % count number of points for average
        end
        
    end
    
    %Normalizing data
    powerPlot = powerPlot/(ntp1 - ntp0 + 1);
    
    allPowerPlot(i_imf,:) = powerPlot(:);
    sPowerPlot = sPowerPlot + powerPlot;
    
end


end





