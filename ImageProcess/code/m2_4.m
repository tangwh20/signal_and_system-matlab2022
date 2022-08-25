%% m2_4.m

clear;
load data\hall.mat hall_gray;

%% DCT2 and IDCT2 transform

hall_gray_A = double(hall_gray)-128; % preprocessing
hall_gray_C = my_dct2(hall_gray_A);

hall_gray_CT = hall_gray_C';
hall_gray_C90 = rot90(hall_gray_C);
hall_gray_C180 = rot90(hall_gray_C90);

hall_gray_T = uint8(my_idct2(hall_gray_CT)+128);
hall_gray_90 = uint8(my_idct2(hall_gray_C90)+128);
hall_gray_180 = uint8(my_idct2(hall_gray_C180)+128);

%% plot

figure;
subplot(2,2,1);
imshow(hall_gray);
title("Original");
subplot(2,2,2);
imshow(hall_gray_T);
title("Transpose");
subplot(2,2,3);
imshow(hall_gray_90);
title("Rotate 90");
subplot(2,2,4);
imshow(hall_gray_180);
title("Rotate 180");
