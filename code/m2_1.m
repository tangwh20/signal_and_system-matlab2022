%% m2_1.m

clear;
load data\hall.mat hall_gray;

%% parameters

N = 8;
A = double(hall_gray(1:8,1:8));
D = dct2_operator(N);

%% transform

C1 = D*A*D';
C1(1,1) = C1(1,1)-128*N;
C2 = D*(A-128)*D';
C1,C2
