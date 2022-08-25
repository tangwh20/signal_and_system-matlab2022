%% m2_13.m

clear;
load data\snow.mat;
[DC_code,AC_code,height,width] = jpeg(snow);
snow_jpeg = dejpeg(DC_code,AC_code,height,width);

%% compression ratio
size_original = height*width*8;
size_compress = strlength(DC_code)+strlength(AC_code);
ratio = size_original/size_compress

%% PSNR
MSE = mean((snow_jpeg-snow).^2,'all');
PSNR = 10*log10(255^2/MSE)

%% plot
figure;
subplot(1,2,1);
imshow(snow);
title("Original");
subplot(1,2,2);
imshow(snow_jpeg);
title("JPEG process");