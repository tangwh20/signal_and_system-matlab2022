%% m2_3.m

clear;
load data\hall.mat hall_gray;

%% DCT2 and IDCT2 transform

hall_gray_A = double(hall_gray)-128; % preprocessing
hall_gray_C = my_dct2(hall_gray_A);

% set right 4 columns zero
hall_gray_C1 = hall_gray_C;
hall_gray_C1(:,end-3:end) = 0;
hall_gray1 = uint8(my_idct2(hall_gray_C1)+128);

% set left 4 columns zero
hall_gray_C2 = hall_gray_C;
hall_gray_C2(:,1:4) = 0;
hall_gray2 = uint8(my_idct2(hall_gray_C2)+128);

%% plot and compare
figure;
subplot(1,3,1);
imshow(hall_gray);
title("Original");
subplot(1,3,2);
imshow(hall_gray1);
title("Set right 4 cols zero");
subplot(1,3,3);
imshow(hall_gray2);
title("Set left 4 cols zero");
