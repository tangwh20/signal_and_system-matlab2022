%% m2_11.m

clear;
load data\jpegcodes.mat;
load data\hall.mat hall_gray;
hall_gray_jpeg = dejpeg(DC_code,AC_code,height,width);

%% quantitative comparison

MSE = mean((hall_gray_jpeg-hall_gray).^2,'all');
PSNR = 10*log10(255^2/MSE)

%% plot

figure;
subplot(1,2,1);
imshow(hall_gray);
title("Original");
subplot(1,2,2);
imshow(hall_gray_jpeg);
title("JPEG process");
