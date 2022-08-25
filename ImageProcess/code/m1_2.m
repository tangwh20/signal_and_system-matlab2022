%% m1_2.m

clear;
load data\hall.mat hall_color;

%% draw circle

len = double(size(hall_color,2));
wid = double(size(hall_color,1));
r = min([len,wid])/2;
ctr = [len/2,wid/2];

figure;
imshow(hall_color);
viscircles(ctr,r);

%% draw black-white cells

cell = uint8([1,0;0,1]);
cell = repmat(cell,wid/2,len/2,3);
hall_color_bw = hall_color.*cell;
figure;
imshow(hall_color_bw);
