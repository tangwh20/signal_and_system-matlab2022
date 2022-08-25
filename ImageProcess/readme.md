# 图像处理大作业

## 一、基础知识

1. MATLAB提供了图像处理工具箱，在命令窗口输入 `help images` 可查看该工具箱内的所有函数。请阅读并大致了解这些函数的基本功能。
2. 利用MATLAB提供的 `Image file I/O` 函数分别完成以下处理：
    1. 以测试图像的中心为圆心，图像的长和宽中较小值的一半为半径画一个红颜色的圆；
    2. 将测试图像涂成国际象棋状的“黑白格”的样子，其中“黑”即黑色，“白”则意味着保留原图。

    用一种看图软件浏览上述两个图，看是否达到了目标。

> 实现代码如下：

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

> 输出分别为


## 二、图像压缩编码

**1. 图像的预处理是将每个像素灰度值减去128，这个步骤是否可以在变换域进行？请在测试图像中截取一块验证你的结论。** 

> 可以在变换域实现。  
>  **直观理解：** 图像每个像素灰度值减去128，相当于从原信号中减去一个直流分量，也就是减去若干倍的零频点强度。对应于DCT算子，即在 $C(0,0)$ 上减去 $128\cdot x$ ( $x$ 待定)。  
>  **数学分析：** 设 $A$ 为 $N\times N$ 的待处理矩阵，记 $D$ 为DCT算子矩阵， $E$ 为全1矩阵。记 $C$ 为减去128之前的变换矩阵, $C’$ 为减去128之后的变换矩阵。  
> 
>> $$ C=DAD^T $$  
>> 
>> $$ C'=D(A-128E)D^T $$
>> 
>> $$ 
>> DED^T=\frac{2}{N}
>> \begin{bmatrix}
>> \frac{1}{\sqrt 2}           &   \frac{1}{\sqrt 2}               &   \cdots  &   \frac{1}{\sqrt 2}               \\
>> \cos \frac{\pi}{2N}         &   \cos \frac{3\pi}{2N}            &   \cdots  &   \frac{(2N-1)\pi}{2N}            \\
>> \vdots                      &   \vdots                          &   \ddots  &   \vdots                          \\
>> \cos \frac{(N-1)\pi}{2N}    &   \cos \frac{3(N-1)\pi}{2N}       &   \cdots  &   \frac{(2N-1)(N-1)\pi}{2N}       
>> \end{bmatrix}
>> \cdot
>> \begin{bmatrix}
>> 1       &   1       &   \cdots  &   1       \\
>> 1       &   1       &   \cdots  &   1       \\
>> \vdots  &   \vdots  &   \ddots  &   \vdots  \\
>> 1       &   1       &   \cdots  &   1       
>> \end{bmatrix}
>> \cdot
>> \begin{bmatrix}
>> \frac{1}{\sqrt 2}   &   \cos \frac{\pi}{2N}     &   \cdots  &   \cos \frac{(N-1)\pi}{2N}    \\
>> \frac{1}{\sqrt 2}   &   \cos \frac{3\pi}{2N}    &   \cdots  &   \cos \frac{3(N-1)\pi}{2N}   \\
>> \vdots              &   \vdots                  &   \ddots  &   \vdots                      \\
>> \frac{1}{\sqrt 2}   &   \frac{(2N-1)\pi}{2N}    &   \cdots  &   \frac{(2N-1)(N-1)\pi}{2N}
>> \end{bmatrix}
>> $$
>> 
>> $$
>> =\frac{2}{N}
>> \begin{bmatrix}
>> \frac{N}{2} &   \frac{N}{2} &   \cdots  &   \frac{N}{2} \\
>> 0           &   0           &   \cdots  &   0           \\
>> \vdots      &   \vdots      &   \ddots  &   \vdots      \\
>> 0           &   0           &   \cdots  &   0       
>> \end{bmatrix}
>> \cdot
>> \begin{bmatrix}
>> \frac{1}{\sqrt 2}   &   \cos \frac{\pi}{2N}     &   \cdots  &   \cos \frac{(N-1)\pi}{2N}    \\
>> \frac{1}{\sqrt 2}   &   \cos \frac{3\pi}{2N}    &   \cdots  &   \cos \frac{3(N-1)\pi}{2N}   \\
>> \vdots              &   \vdots                  &   \ddots  &   \vdots                      \\
>> \frac{1}{\sqrt 2}   &   \frac{(2N-1)\pi}{2N}    &   \cdots  &   \frac{(2N-1)(N-1)\pi}{2N}
>> \end{bmatrix}=
>> \begin{bmatrix}
>> N       &   0       &   \cdots  &   0       \\
>> 0       &   0       &   \cdots  &   0       \\
>> \vdots  &   \vdots  &   \ddots  &   \vdots  \\
>> 0       &   0       &   \cdots  &   0       
>> \end{bmatrix}
>> $$
>>  
>> $$ 
>> C'=C-128DED^T=C-
>> \begin{bmatrix}
>> 128N    &   0       &   \cdots  &   0       \\
>> 0       &   0       &   \cdots  &   0       \\
>> \vdots  &   \vdots  &   \ddots  &   \vdots  \\
>> 0       &   0       &   \cdots  &   0       
>> \end{bmatrix}
>> $$
>
> 故数学验证成立  

> 代码实现如下：

```matlab
%% m2_1.m

clear;
load data\hall.mat hall_gray;

%% parameters

N = 8;
A = double(hall_gray(1:8,1:8));
v1 = 1:2:2*N-1;
v2 = (0:N-1)';
D = sqrt(2/N)*cos(kron(v1,v2)*pi/2/N);
D(1,:) = D(1,:)/sqrt(2);

%% transform

C1 = D*A*D';
C1(1,1) = C1(1,1)-128*N;
C2 = D*(A-128)*D';
C1,C2
```

> 输出结果如下：
> 
> 可以发现两种方法得到的图像完全相同，即验证了上述结论。  
  
**2. 请编程实现二维DCT，并和MATLAB自带的库函数dct2比较是否一致。** 
> 
> 实现二维DCT只需要实现二维DCT算子 $D$ ，之后直接代入公式
>
>> $$ C_{M\times N}=D_{M\times M}A_{M\times N}D^T_{N\times N} $$
> 
> 由1可得，DCT算子
> 
>> $$ D_{N\times N}=
>> \frac{2}{N}
>> \begin{bmatrix}
>> \frac{1}{\sqrt 2}           &   \frac{1}{\sqrt 2}               &   \cdots  &   \frac{1}{\sqrt 2}               \\
>> \cos \frac{\pi}{2N}         &   \cos \frac{3\pi}{2N}            &   \cdots  &   \frac{(2N-1)\pi}{2N}            \\
>> \vdots                      &   \vdots                          &   \ddots  &   \vdots                          \\
>> \cos \frac{(N-1)\pi}{2N}    &   \cos \frac{3(N-1)\pi}{2N}       &   \cdots  &   \frac{(2N-1)(N-1)\pi}{2N}       
>> \end{bmatrix}
>> $$  
> 
> 构造DCT算子的生成函数 `dct2_operator.m` 和二维DCT的实现函数 `my_dct2.m`  

```matlab
function D = dct2_operator(N)
%DCT2_OPERATOR operator matrix D of 2-dimensional DCT transform
% This function returns N*N matrix D based on the input number N

v1 = 1:2:2*N-1;
v2 = (0:N-1)';
D = sqrt(2/N)*cos(kron(v1,v2)*pi/2/N);
D(1,:) = D(1,:)/sqrt(2);

end
```

```matlab
function C = my_dct2(A)
%MY_DCT2 2-dimensional DCT transform
% This function returns the 2-dimensional DCT transform of the input
% matrix A

[m,n] = size(A);
Dm = dct2_operator(m);
Dn = dct2_operator(n);
C = Dm*A*Dn';

end
```

> 验证功能：

```matlab
%% m2_2.m

clear;

%% generate random matrix for test

sz = ceil(10*rand(1,2));
A = rand(sz);
C1 = dct2(A)
C2 = my_dct2(A)
```

> 输出结果如下：  
> 可见 $C1$ 与 $C2$ 完全相同，即验证了自行实现的二维DCT变换的正确性。  

**3. 如果将DCT系数矩阵中右侧四列的系数全部置零，逆变换后的图像会发生什么变化？选取一块图验证你的结论。如果左侧的四列置零呢？**

> 首先仿照 `my_dct2` 函数的定义方法自定义 `my_idct2` 函数进行逆DCT变换

```matlab
function A = my_idct2(C)
%MY_IDCT2 2-dimensional inverse DCT transform
% This function returns the 2-dimensional inverse DCT transform of the 
% input matrix C

[m,n] = size(C);
Dm = dct2_operator(m);
Dn = dct2_operator(n);
A = Dm'*C*Dn;

end
```

> 选取hall_gray图片进行处理
> 实现代码如下：

```matlab
%% m2_3.m

clear;
load data\hall.mat hall_gray;

%% DCT2 and IDCT2 transform
hall_gray_A = double(hall_gray)-128; % preprocessing
hall_gray_C = my_dct2(hall_gray_A);
% set right 4 columns zero
hall_gray_C1 = hall_gray_C;
hall_gray_C1(:,end-3:end) = 0;
hall_gray1 = uint8(my_idct2(hall_gray_C1)+128);
% set left 4 columns zero
hall_gray_C2 = hall_gray_C;
hall_gray_C2(:,1:4) = 0;
hall_gray2 = uint8(my_idct2(hall_gray_C2)+128);

%% plot and compare
figure;
subplot(1,3,1);
imshow(hall_gray);
title("Original");
subplot(1,3,2);
imshow(hall_gray1);
title("Set right 4 cols zero");
subplot(1,3,3);
imshow(hall_gray2);
title("Set left 4 cols zero");
```

> 输出如下：  
> 可见，将右侧4列（高频分量）置零后，图像几乎无变化；将左侧4列（直流和低频分量）置零后，图像明显变暗。因此可以说明，人眼对图像中的低频分量比高频分量更敏感。另外，由于将直流分量也置为零，因此图像明显变暗（灰度数值整体减小）。

** 4．若对DCT系数分别做转置、旋转90度和旋转180度操作 (rot90) ，逆变换后恢复的图像有何变化？选取一块图验证你的结论。** 

> 选取hall_gray图片进行处理：

```matlab
%% m2_4.m

clear;
load data\hall.mat hall_gray;

%% DCT2 and IDCT2 transform

hall_gray_A = double(hall_gray)-128; % preprocessing
hall_gray_C = my_dct2(hall_gray_A);

hall_gray_CT = hall_gray_C';
hall_gray_C90 = rot90(hall_gray_C);
hall_gray_C180 = rot90(hall_gray_C90);

hall_gray_T = uint8(my_idct2(hall_gray_CT)+128);
hall_gray_90 = uint8(my_idct2(hall_gray_C90)+128);
hall_gray_180 = uint8(my_idct2(hall_gray_C180)+128);

%% plot
figure;
subplot(2,2,1);
imshow(hall_gray);
title("Original");
subplot(2,2,2);
imshow(hall_gray_T);
title("Transpose");
subplot(2,2,3);
imshow(hall_gray_90);
title("Rotate 90");
subplot(2,2,4);
imshow(hall_gray_180);
title("Rotate 180");
```

> 输出如下：  
> 可见，DCT系数矩阵转置后图片仅仅是旋转了90°；DCT系数矩阵旋转90°后，图片不仅旋转了90°，还产生了许多黑白条纹，大礼堂的形状变得模糊；DCT系数矩阵旋转180°后，图片并没有旋转，但产生了大量黑白点，形状变得特别模糊。

** 5．如果认为差分编码是一个系统，请绘出这个系统的频率响应，说明它是一个*高通* (低通、高通、带通、带阻) 滤波器。DC系数先进行差分编码再进行熵编码，说明DC系数的 *高通* 频率分量更多。** 

> 差分编码的表达式为 $ y(n)=x(n-1)-x(n) $
> 传递函数为
> $$ H(s)=-\frac{1}{1-s^-1} $$
> 
