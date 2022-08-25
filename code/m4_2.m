%% m4_2.m

clear;
img = imread("data\image2.jpg");
res = testFeature(img,32,5,0.7);

%% create boundary
B = bwboundaries(res);

%% plot
f = figure;
imshow(img);
hold on;
visboundaries(B);
saveas(f,"data\image2_proc.jpg");