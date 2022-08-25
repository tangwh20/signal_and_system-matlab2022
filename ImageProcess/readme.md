# 图像处理大作业
## 一、基础知识

1. MATLAB提供了图像处理工具箱，在命令窗口输入help images可查看该工具箱内的所有函数。请阅读并大致了解这些函数的基本功能。
2. 利用MATLAB 提供的Image file I/O函数分别完成以下处理：
    1. 以测试图像的中心为圆心，图像的长和宽中较小值的一半为半径画一个红颜色的圆；
    2. 将测试图像涂成国际象棋状的“黑白格”的样子，其中“黑”即黑色，“白”则意味着保留原图。

    用一种看图软件浏览上述两个图，看是否达到了目标。
    
```matlab
%% m1_2.m

clear;
load data\hall.mat hall_color;

%% draw circle

len = double(size(hall_gray,2));
wid = double(size(hall_gray,1));
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
```

输出分别为


## 二、图像压缩编码

