%% m3_2.m

clear;
load data\hall.mat hall_gray;
figure;
subplot(2,2,1);
imshow(hall_gray);
title("hall");

%% info process
info = ["";"";"Peking University"];
info_extract = strings(3,1);
for i=1:49
    info(1) = info(1)+"Peking University is the best university in China."+newline;
end
for i=1:24
    info(2) = info(2)+"Peking University is the best university in China."+newline;
end

ratio_pku = ones(1,3);
PSNR_pku = ones(1,3);
info_correct = zeros(1,3);
for i=1:3
    [hall_gray_pku,DC_code_pku,AC_code_pku] = dctHide(hall_gray,info(i),i);
    info_extract(i) = dctExtract(hall_gray_pku,i);
    info_correct(i) = isequal(info(i),info_extract(i));
    
    %% compression ratio and PSNR
    load data\jpegcodes.mat;
    size_original = height*width*8;
    size_compress = strlength(DC_code)+strlength(AC_code);
    ratio = size_original/size_compress;
    size_compress_pku = strlength(DC_code_pku)+strlength(AC_code_pku);
    ratio_pku(i) = size_original/size_compress_pku;
    MSE_pku = mean((uint8(hall_gray_pku)-hall_gray).^2,'all');
    PSNR_pku(i) = 10*log10(255^2/MSE_pku);

    %% plot    
    subplot(2,2,i+1);
    imshow(uint8(hall_gray_pku));
    title("hall\_pku"+string(i));
end

ratio,ratio_pku,PSNR_pku,info_correct