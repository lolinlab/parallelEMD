% pure_EMDcuda+env2.cu performes EMD on each of the column of the input
% with parallel computing on GPU
% Copyright (C):
% (1) Medical Biodynamics Program, Brigham and Women's Hospital, 
% (2) Lab of Integrated Biosignal Advances, National Central University; 2022
% ***** For Educational and Academic Purposes Only ********* < Ask 則余
% Orignal Authors: Li-Wen Chang, Men-Tzung Lo, Nasser Anssari, Ke-Hsin Hsu, Norden E. Huang, Wen-mei W. Hwu
% Modifying Authors: Yu-Chi Peng, Hui-Wen Yang, Jin-En Hsu, Yi-Je Lin, Kun Hu, Men-Tzung Lo
% Please Cite: 
% Li-Wen Chang, Men-Tzung Lo, Nasser Anssari, Ke-Hsin Hsu, Norden E. Huang, Wen-mei W. Hwu
% "PARALLEL IMPLEMENTATION OF MULTI-DIMENSIONAL ENSEMBLE EMPIRICAL MODE DECOMPOSITION"
% ICASSP 2011
% 
% Usage: 
% [out0,val0,idx0,len0,up0] = mex_pureEMD_env(A,nm, nsift);
% 
% input:
% (1) A: matrix (x_len by y_len), each column is the signal for 1-D EMD
% (2) nm: number of modes desired
% (3) nsift: number of sifting, please use 10. (will set as default in
% future version)
% 
% output: 
% (1) out0 (x_len*y_len*nm by 1): IMFs of all the input, need to reshape
% (2) val0 (2*x_len*y_len*nm by 1): values of critical points of the IMFs
% (3) idx0 (2*x_len*y_len*nm by 1): indexes of critical points of the IMFs
% (4) len0 (y_len by nm ): number of critical points of the IMFs
% (5) up0 (x_len*y_len*nm by 1): Upper envelop of all the IMFs, need to reshape
% 

clc; clear; close all;
addpath('../cuda')
%% For the first time use, uncomment this line to compile 
% mexcuda mex_gpuEMD_env.cu
%% Generate signal
%A = [1.5, 2, 0.2; 2.2, -3.3 1.0]
%A = rand(4,20);
x_len = 390; y_len = 203; nm = 3; nsift = 10;
fs = 200;
t = (1:x_len)/fs;
x0 = zeros(x_len,nm);
for k = 1:nm
    f = 3.3^(nm+1-k);
    amp = (1.8)^(k);
    x0(:,k) = amp*sin(2*pi*f*t)';
end
x = sum(x0,2);
A = repmat(x,1,y_len); % replication

%% Decompose by parallel EMD
nm = 2;
[out0,val0,idx0,len0,up0] = mex_gpuEMD_env(A,nm, nsift);

out = reshape(out0, x_len, y_len, nm);
val = reshape(val0, x_len*2, y_len, nm);
idx = reshape(idx0, x_len*2, y_len, nm);
len = reshape(len0, y_len, nm);
up = reshape(up0, x_len, y_len, nm);

%%
% plot the figure
figure;
hold off;
subplot(nm+1,1,1);
plot(A(:,1),'k','LineWidth',2);
hold on;
plot(sum(out(:,1,:),3),'r','LineWidth',1);
yi = 1;
for i = 1:nm
    subplot(nm+1,1,i+1);
    dname = ['IMF ' num2str(i)];
    plot(out(:,yi,i),'DisplayName',dname);hold on;
    plot(up(:,yi,i),'DisplayName','Inst. amp');
    plot(idx(1:len(yi,i),yi,i)+1,val(1:len(yi,i),yi,i),'o','DisplayName','cpts');
    legend()
end
