%% m3_1.m

clear;
load data\hall.mat hall_gray;

%% info process
info = "";
for i=1:49
    info = info+"Peking University is the best university in China."+newline;
end
hall_gray_pku = spatialHide(hall_gray,info); % hide info
info_extract = spatialExtract(hall_gray_pku) % extract info
[DC_code,AC_code,height,width] = jpeg(hall_gray_pku); % jpeg encoding
hall_gray_pku_jpeg = dejpeg(DC_code,AC_code,height,width); % jpeg decoding
info_extract_jpeg = spatialExtract(hall_gray_pku_jpeg) % extract from jpeg

%% plot
figure;
subplot(1,3,1);
imshow(hall_gray);
title("hall");
subplot(1,3,2);
imshow(hall_gray_pku);
title("hall\_pku");
subplot(1,3,3);
imshow(hall_gray_pku_jpeg);
title("hall\_pku\_jpeg");
