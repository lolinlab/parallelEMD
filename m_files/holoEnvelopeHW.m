%This code was copied from multi_EMD.m, found in holospectrum\Holo Spect\,
%code written by Dr. Lo's group. Only the name has been changed.
%This copy created on July 17, 2020.
%Update July 1, 2021 use indmin, indmax from above if passed in
%%
function [envmax,envmin,tmax,tmin,xmax,xmin envmax0, envmin0, indmax, indmin] = holoEnvelopeHW(data,INTERP,indmin,indmax,indzer)
%computes envelopes and mean with various interpolations

NBSYM = 2;
DEF_INTERP = 'spline';
t = 1:length(data);%Update 6-30-2021 moved outside if nargin <2 because should always happen

if nargin < 2
    INTERP = DEF_INTERP;
end

if ~ischar(INTERP)
    error('interp parameter must be ''linear'''', ''cubic'' or ''spline''')
end

if ~any(strcmpi(INTERP,{'linear','cubic','spline'}))
    error('interp parameter must be ''linear'''', ''cubic'' or ''spline''')
end

if (nargin < 4)
    calcExtr = true;
else
    calcExtr = false;
end

s = size(data);
if s(1) > s(2)
    data = data';
end

for ijk = 1:size(data,1)
    x = data(ijk,:);
    lx = length(x);
    if calcExtr
        [indmin,indmax,indzer] = extr(x,t);
    end
        
    
    if (length(indmin) + length(indmax) < 3)
      %  error('not enough extrema')
        envmax(ijk,:) = abs(x);
        envmin(ijk,:) = -abs(x);
    else
        envmax0(ijk,:) = interp1(indmax,x(indmax),t,INTERP);
        envmin0(ijk,:) = interp1(indmin,x(indmin),t,INTERP);
        
        %boundary conditions for interpolation

        [tmin,tmax,xmin,xmax] = boundary_conditions(indmin,indmax,t,x,NBSYM);
        
        % definition of envelopes from interpolation
        envmax(ijk,:) = interp1(tmax,xmax,t,INTERP);
        envmin(ijk,:) = interp1(tmin,xmin,t,INTERP);
    end
end
if s(1) > s(2)
    envmax = envmax';
    envmin = envmin';
end
%---------------------------------------------------------------------------------------

function [tmin,tmax,xmin,xmax] = boundary_conditions(indmin,indmax,t,x,nbsym)
% computes the boundary conditions for interpolation (mainly mirror symmetry)


lx = length(x);

if (length(indmin) + length(indmax) < 3)
    error('not enough extrema')
end

if indmax(1) < indmin(1)
    if x(1) > x(indmin(1))
        lmax = fliplr(indmax(2:min(end,nbsym+1)));
        lmin = fliplr(indmin(1:min(end,nbsym)));
        lsym = indmax(1);
    else
        lmax = fliplr(indmax(1:min(end,nbsym)));
        lmin = [fliplr(indmin(1:min(end,nbsym-1))),1];
        lsym = 1;
    end
else
    
    if x(1) < x(indmax(1))
        lmax = fliplr(indmax(1:min(end,nbsym)));
        lmin = fliplr(indmin(2:min(end,nbsym+1)));
        lsym = indmin(1);
    else
        lmax = [fliplr(indmax(1:min(end,nbsym-1))),1];
        lmin = fliplr(indmin(1:min(end,nbsym)));
        lsym = 1;
    end
end

if indmax(end) < indmin(end)
    if x(end) < x(indmax(end))
        rmax = fliplr(indmax(max(end-nbsym+1,1):end));
        rmin = fliplr(indmin(max(end-nbsym,1):end-1));
        rsym = indmin(end);
    else
        rmax = [lx,fliplr(indmax(max(end-nbsym+2,1):end))];
        rmin = fliplr(indmin(max(end-nbsym+1,1):end));
        rsym = lx;
    end
else
    if x(end) > x(indmin(end))
        rmax = fliplr(indmax(max(end-nbsym,1):end-1));
        rmin = fliplr(indmin(max(end-nbsym+1,1):end));
        rsym = indmax(end);
    else
        rmax = fliplr(indmax(max(end-nbsym+1,1):end));
        rmin = [lx,fliplr(indmin(max(end-nbsym+2,1):end))];
        rsym = lx;
    end
end

tlmin = 2*t(lsym)-t(lmin);
tlmax = 2*t(lsym)-t(lmax);
trmin = 2*t(rsym)-t(rmin);
trmax = 2*t(rsym)-t(rmax);

% in case symmetrized parts do not extend enough
if tlmin(1) > t(1) | tlmax(1) > t(1)
    if lsym == indmax(1)
        lmax = fliplr(indmax(1:min(end,nbsym)));
    else
        lmin = fliplr(indmin(1:min(end,nbsym)));
    end
    if lsym == 1
        error('bug')
    end
    lsym = 1;
    tlmin = 2*t(lsym)-t(lmin);
    tlmax = 2*t(lsym)-t(lmax);
end

if trmin(end) < t(lx) | trmax(end) < t(lx)
    if rsym == indmax(end)
        rmax = fliplr(indmax(max(end-nbsym+1,1):end));
    else
        rmin = fliplr(indmin(max(end-nbsym+1,1):end));
    end
    if rsym == lx
        error('bug')
    end
    rsym = lx;
    trmin = 2*t(rsym)-t(rmin);
    trmax = 2*t(rsym)-t(rmax);
end

xlmax =x(lmax);
xlmin =x(lmin);
xrmax =x(rmax);
xrmin =x(rmin);

tmin = [tlmin t(indmin) trmin];
tmax = [tlmax t(indmax) trmax];
xmin = [xlmin x(indmin) xrmin];
xmax = [xlmax x(indmax) xrmax];

%---------------------------------------------------------------------------------------------------

function [indmin, indmax, indzer] = extr(x,t);
%extracts the indices corresponding to extrema

if(nargin==1)
    t=1:length(x);
end
if size(x,1)>size(x,2)
    x=x';
end
m = length(x);

if nargout > 2
    x1=x(1:m-1);
    x2=x(2:m);
    indzer = find(x1.*x2<0);
    
    if any(x == 0)
        iz = find( x==0 );
        indz = [];
        if any(diff(iz)==1)
            zer = x == 0;
            dz = diff([0 zer 0]);
            debz = find(dz == 1);
            finz = find(dz == -1)-1;
            indz = round((debz+finz)/2);
        else
            indz = iz;
        end
        indzer = sort([indzer indz]);
    end
end

d = diff(x);

n = length(d);
d1 = d(1:n-1);
d2 = d(2:n);
indmin = find(d1.*d2<0 & d1<0)+1;
indmax = find(d1.*d2<0 & d1>0)+1;


% when two or more consecutive points have the same value we consider only one extremum in the middle of the constant area

if any(d==0)
    
    imax = [];
    imin = [];
    
    bad = (d==0);
    dd = diff([0 bad 0]);
    debs = find(dd == 1);
    fins = find(dd == -1);
    if debs(1) == 1
        if length(debs) > 1
            debs = debs(2:end);
            fins = fins(2:end);
        else
            debs = [];
            fins = [];
        end
    end
    if length(debs) > 0
        if fins(end) == m
            if length(debs) > 1
                debs = debs(1:(end-1));
                fins = fins(1:(end-1));
                
            else
                debs = [];
                fins = [];
            end
        end
    end
    lc = length(debs);
    if lc > 0
        for k = 1:lc
            if d(debs(k)-1) > 0
                if d(fins(k)) < 0
                    imax = [imax round((fins(k)+debs(k))/2)];
                end
            else
                if d(fins(k)) > 0
                    imin = [imin round((fins(k)+debs(k))/2)];
                end
            end
        end
    end
    
    if length(imax) > 0
        indmax = sort([indmax imax]);
    end
    
    if length(imin) > 0
        indmin = sort([indmin imin]);
    end
    
end

