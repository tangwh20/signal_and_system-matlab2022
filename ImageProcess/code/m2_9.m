%% m2_9.m

clear;
load data\hall.mat hall_gray;
[DC_code,AC_code,height,width] = jpeg(hall_gray);
save data\jpegcodes DC_code AC_code height width;