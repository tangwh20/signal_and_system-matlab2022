%% m4_3.m

clear;
img = imread("data\image2.jpg");
img_rot = imrotate(img,-90);
img_ext = imresize(img,'Scale',[1,2]);
img_adj = imadjust(img,[.2 .3 0; .6 .7 1],[]);

%% process
res = testFeature(img,32,5,0.7);
res_rot = testFeature(img_rot,32,5,0.7);
res_ext = testFeature(img_ext,32,5,0.72);
res_adj = testFeature(img_adj,32,5,0.7);
B = bwboundaries(res);
B_rot = bwboundaries(res_rot);
B_ext = bwboundaries(res_ext);
B_adj = bwboundaries(res_adj);

%% plot
figure;

% imshow(img);
% hold on;
% visboundaries(B);
% title("Original");
% 
% imshow(img_rot);
% hold on;
% visboundaries(B_rot);
% title("Rotate 90");

imshow(img_ext);
hold on;
visboundaries(B_ext);
title("Resize [1,2]");

% imshow(img_adj);
% hold on;
% visboundaries(B_adj);
% title("Adjust color");

