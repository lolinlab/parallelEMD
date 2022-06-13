%function [f]=FAzc(data,dt)
%
% The function FAzc generates a frequency and amplitude using zero-crossing method
% applied to data(n,k), where n specifies the length of time series,
% and k is the number of IMFs.
% Non MATLAB Library routine used in the function is: FINDCRITICALPOINTS.
%
% Calling sequence-
% [f,a]=fazc(data,dt)
%
% Input-
%	criticalpt_index	- 1-D matrix of critical point
%	idt	    - inverse of time increment per point (frequency)
% Output-
%	f	    - 2-D matrix f(n,k) that specifies frequency
%
% Used by-
%	FA
% See also-
% 	FAZPLUS, which in addition to frequency and amplitude, outputs
%	other fields.

%written by
% Kenneth Arnold (NASA GSFC)	Summer 2003, Modified
% Xianyao Chen     Sep. 20 created following the splinenormalize.m
% S.C.Su Sep. 2009(NCU Rcada)rename faz.m as fazc.m
%  in order to integrate all m file in a group type
%%footnote: S.C.Su (2009/09/07)
% Modified by Hui-Wen  2021-01-08
%
% There are two loops in this code ,it's dealing with IMF by IMF .checking wave by wave
%  0. set the default value and initialize
%  1. process IMF by IMF ---loop A start
%    2. Find all critical points (max,min,zeros) in an IMF
%    3. integrate those information from previous waveform
%    4. calculate by zero-crossing method----loop B start
%            4.1 estimate for the quarter waves(weighting coefficients included already)
%            4.2 estimate for the half waves(weighting coefficients included already)
%            4.3 estimate for the whole waves(weighting coefficients included already)
%            5. add-in those information from previous waveform
%            6. calculate current frequency and amplitude
%            7. pass current information to next waveform calculation
%    4. calculate by zero-crossing method----loop B end
%  1. process IMF by IMF ---loop A end
%
%

%function [f]=FAzc(data,dt)
function [f]=FAzc2(allX,nPoints,idt)
% allX is the critical points
% nPoints is the length of the real signal;
% idt is the inverse of time increment (sample frequency)

% 0. set the default value and initialize
%----- Get dimensions
sdim = size(allX);
if sdim(1) > sdim(2), allX = allX'; end
%----- Flip data if necessary
flipped=0;
if min(sdim) > 1
    print('input for FAzc2 should be a vector');
    return
end


%----- Inverse dt
%idt = 1/dt; idt is the frequency

%----- Preallocate arrays
f = zeros(nPoints,1);

%1. process IMF by IMF ---loop A start
%----- Process each IMF
%for c=1:nIMF   %loop A--start

%2. Find all critical points (max,min,zeros) in an IMF
%The function FINDCRITICALPOINTS returns max, min and zero crossing values and their coordinates in the order found
%[allX, allY] = findcriticalpoints(data(:,c));
nCrit = length(allX); %number of critical points

%     if nCrit <= 1
%         %----- Too few critical points; keep looping
%         continue;
%     end


%%

f1 = diff(allX);
f2 = allX(3:nCrit) - allX(1:nCrit-2);
f2cur = [f2, NaN];
f2pre = [NaN, f2];
f4 = allX(5:nCrit) - allX(1:nCrit-4);
f4cur = [f4, nan, nan, nan];
f4prev3 = [nan, f4, nan, nan];
f4prev2 = [nan, nan, f4, nan];
f4prev1 = [nan, nan, nan, f4];

allf = cat(1, f1,f2cur, f2pre, f4cur, f4prev1, f4prev2, f4prev3);
allf = 1./allf;
allnpt = ~isnan(allf);
w = [4,2,2,1,1,1,1];
allw = w*allnpt;
ftotal2 = idt*nansum(allf,1)./allw;

for i = 1:nCrit-1
    f(ceil(allX(i)):floor(allX(i+1))) = ftotal2(i);
end
f(1:ceil(allX(1))-1) = f(ceil(allX(1)));
f(floor(allX(nCrit))+1:nPoints) = f(floor(allX(nCrit)));
