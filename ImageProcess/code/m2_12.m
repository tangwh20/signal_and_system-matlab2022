%% m2_12.m

clear;
load data\hall.mat hall_gray;
[DC_code,AC_code,height,width] = jpeg(hall_gray);
hall_gray_jpeg = dejpeg(DC_code,AC_code,height,width);

%% compression ratio
size_original = height*width*8;
size_compress = strlength(DC_code)+strlength(AC_code);
ratio = size_original/size_compress

%% PSNR
MSE = mean((hall_gray_jpeg-hall_gray).^2,'all');
PSNR = 10*log10(255^2/MSE)

%% plot
figure;
subplot(1,2,1);
imshow(hall_gray);
title("Original");
subplot(1,2,2);
imshow(hall_gray_jpeg);
title("JPEG process 2");