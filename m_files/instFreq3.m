function [InsFreq1] = instFreq3(criticalPoints,nPoints, sampFreq, nCrit)
%Since fa doesn't do much (more robust and used for testing different
%methods, but we only want the zero crossing frequency, rewrote the
%lines that are actually used here
% criticalPoints: columnwise


InsFreq1 = zeros(nPoints, size(criticalPoints,2));
for icol = 1:size(criticalPoints,2)
    if nCrit(icol) < 6, continue; end
    f = FAzc2(criticalPoints(1:nCrit(icol),icol),nPoints,sampFreq);
    InsFreq1(:,icol) = f';
end