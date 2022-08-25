%% m2_10.m

clear;
load data\jpegcodes.mat;
size_original = height*width*8
size_compress = strlength(DC_code)+strlength(AC_code)
ratio = size_original/size_compress
