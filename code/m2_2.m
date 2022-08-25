%% m2_2.m

clear;

%% generate random matrix for test

sz = ceil(10*rand(1,2));
A = rand(sz);
C1 = dct2(A)
C2 = my_dct2(A)