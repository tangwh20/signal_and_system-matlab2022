# 图像处理大作业

## 一、基础知识

> 1. MATLAB提供了图像处理工具箱，在命令窗口输入 `help images` 可查看该工具箱内的所有函数。请阅读并大致了解这些函数的基本功能。
> 2. 利用MATLAB提供的 `Image file I/O` 函数分别完成以下处理：
>     1. 以测试图像的中心为圆心，图像的长和宽中较小值的一半为半径画一个红颜色的圆；
>     2. 将测试图像涂成国际象棋状的“黑白格”的样子，其中“黑”即黑色，“白”则意味着保留原图。
> 
>     用一种看图软件浏览上述两个图，看是否达到了目标。

实现代码如下：

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

&nbsp;

## 二、图像压缩编码

&nbsp;
> 1. 图像的预处理是将每个像素灰度值减去128，这个步骤是否可以在变换域进行？请在测试图像中截取一块验证你的结论。

可以在变换域实现。  
**直观理解：** 图像每个像素灰度值减去128，相当于从原信号中减去一个直流分量，也就是减去若干倍的零频点强度。对应于DCT算子，即在 $C(0,0)$ 上减去 $128\cdot x$ ( $x$ 待定)。  
**数学分析：** 设 $A$ 为 $N\times N$ 的待处理矩阵，记 $D$ 为DCT算子矩阵， $E$ 为全1矩阵。记 $C$ 为减去128之前的变换矩阵, $C’$ 为减去128之后的变换矩阵。  

$$ C=DAD^T $$  

$$ C'=D(A-128E)D^T $$

$$ 
DED^T=\frac{2}{N}
\begin{bmatrix}
\frac{1}{\sqrt 2}           &   \frac{1}{\sqrt 2}               &   \cdots  &   \frac{1}{\sqrt 2}               \\
\cos \frac{\pi}{2N}         &   \cos \frac{3\pi}{2N}            &   \cdots  &   \frac{(2N-1)\pi}{2N}            \\
\vdots                      &   \vdots                          &   \ddots  &   \vdots                          \\
\cos \frac{(N-1)\pi}{2N}    &   \cos \frac{3(N-1)\pi}{2N}       &   \cdots  &   \frac{(2N-1)(N-1)\pi}{2N}       
\end{bmatrix}
\cdot
\begin{bmatrix}
1       &   1       &   \cdots  &   1       \\
1       &   1       &   \cdots  &   1       \\
\vdots  &   \vdots  &   \ddots  &   \vdots  \\
1       &   1       &   \cdots  &   1       
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{1}{\sqrt 2}   &   \cos \frac{\pi}{2N}     &   \cdots  &   \cos \frac{(N-1)\pi}{2N}    \\
\frac{1}{\sqrt 2}   &   \cos \frac{3\pi}{2N}    &   \cdots  &   \cos \frac{3(N-1)\pi}{2N}   \\
\vdots              &   \vdots                  &   \ddots  &   \vdots                      \\
\frac{1}{\sqrt 2}   &   \frac{(2N-1)\pi}{2N}    &   \cdots  &   \frac{(2N-1)(N-1)\pi}{2N}
\end{bmatrix}
$$

$$
=\frac{2}{N}
\begin{bmatrix}
\frac{N}{2} &   \frac{N}{2} &   \cdots  &   \frac{N}{2} \\
0           &   0           &   \cdots  &   0           \\
\vdots      &   \vdots      &   \ddots  &   \vdots      \\
0           &   0           &   \cdots  &   0       
\end{bmatrix}
\cdot
\begin{bmatrix}
\frac{1}{\sqrt 2}   &   \cos \frac{\pi}{2N}     &   \cdots  &   \cos \frac{(N-1)\pi}{2N}    \\
\frac{1}{\sqrt 2}   &   \cos \frac{3\pi}{2N}    &   \cdots  &   \cos \frac{3(N-1)\pi}{2N}   \\
\vdots              &   \vdots                  &   \ddots  &   \vdots                      \\
\frac{1}{\sqrt 2}   &   \frac{(2N-1)\pi}{2N}    &   \cdots  &   \frac{(2N-1)(N-1)\pi}{2N}
\end{bmatrix}=
\begin{bmatrix}
N       &   0       &   \cdots  &   0       \\
0       &   0       &   \cdots  &   0       \\
\vdots  &   \vdots  &   \ddots  &   \vdots  \\
0       &   0       &   \cdots  &   0       
\end{bmatrix}
$$
 
$$ 
C'=C-128DED^T=C-
\begin{bmatrix}
128N    &   0       &   \cdots  &   0       \\
0       &   0       &   \cdots  &   0       \\
\vdots  &   \vdots  &   \ddots  &   \vdots  \\
0       &   0       &   \cdots  &   0       
\end{bmatrix}
$$

故数学验证成立  
代码实现如下：
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

输出结果如下：

可以发现两种方法得到的图像完全相同，即验证了上述结论。

&nbsp;
  
> 2. 请编程实现二维DCT，并和MATLAB自带的库函数dct2比较是否一致。

实现二维DCT只需要实现二维DCT算子 $D$ ，之后直接代入公式

$$ C_{M\times N}=D_{M\times M}A_{M\times N}D^T_{N\times N} $$

由1可得，DCT算子

$$ D_{N\times N}=
\frac{2}{N}
\begin{bmatrix}
\frac{1}{\sqrt 2}           &   \frac{1}{\sqrt 2}               &   \cdots  &   \frac{1}{\sqrt 2}               \\
\cos \frac{\pi}{2N}         &   \cos \frac{3\pi}{2N}            &   \cdots  &   \frac{(2N-1)\pi}{2N}            \\
\vdots                      &   \vdots                          &   \ddots  &   \vdots                          \\
\cos \frac{(N-1)\pi}{2N}    &   \cos \frac{3(N-1)\pi}{2N}       &   \cdots  &   \frac{(2N-1)(N-1)\pi}{2N}       
\end{bmatrix}
$$  

构造DCT算子的生成函数 `dct2_operator.m` 和二维DCT的实现函数 `my_dct2.m`  

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

验证功能：

```matlab
%% m2_2.m

clear;

%% generate random matrix for test

sz = ceil(10*rand(1,2));
A = rand(sz);
C1 = dct2(A)
C2 = my_dct2(A)
```

输出结果如下：  
可见 $C1$ 与 $C2$ 完全相同，即验证了自行实现的二维DCT变换的正确性。  

&nbsp;

> 3. 如果将DCT系数矩阵中右侧四列的系数全部置零，逆变换后的图像会发生什么变化？选取一块图验证你的结论。如果左侧的四列置零呢？

首先仿照 `my_dct2` 函数的定义方法自定义 `my_idct2` 函数进行逆DCT变换

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

选取hall_gray图片进行处理
实现代码如下：

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

输出如下：  
可见，将右侧4列（高频分量）置零后，图像几乎无变化；将左侧4列（直流和低频分量）置零后，图像明显变暗。因此可以说明，人眼对图像中的低频分量比高频分量更敏感。另外，由于将直流分量也置为零，因此图像明显变暗（灰度数值整体减小）。

&nbsp;

> 4. 若对DCT系数分别做转置、旋转90度和旋转180度操作 (rot90) ，逆变换后恢复的图像有何变化？选取一块图验证你的结论。

选取hall_gray图片进行处理：

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

输出如下：  
可见，DCT系数矩阵转置后图片仅仅是旋转了90°；DCT系数矩阵旋转90°后，图片不仅旋转了90°，还产生了许多黑白条纹，大礼堂的形状变得模糊；DCT系数矩阵旋转180°后，图片并没有旋转，但产生了大量黑白点，形状变得特别模糊。

&nbsp;

> 5. 如果认为差分编码是一个系统，请绘出这个系统的频率响应，说明它是一个*高通* (低通、高通、带通、带阻) 滤波器。DC系数先进行差分编码再进行熵编码，说明DC系数的 *高通* 频率分量更多。

差分编码的表达式为 $y(n)=x(n-1)-x(n)$  
传递函数为

$$ H(s)=-\frac{1}{1-s^-1} $$

代码如下：

```matlab
%% m2_5.m

clear;
a = -1;
b = [1,-1];
freqz(b,a);
```

输出如图所示，可以发现差分编码是一个高通系统。  
DC系数先进行差分编码再进行熵编码，故高频分量更多。

&nbsp;

> 6. DC预测误差的取值和Category值有何关系？如何利用预测误差计算出其 Category？

根据Category表  
可得，Category数n与预测误差x的关系为

$$ n=
\begin{cases}
\lfloor \log_2 |x| \rfloor	&	x\neq 0	\\
0							&	x=0
\end{cases}
$$

&nbsp;

> 7. 你知道哪些实现Zig-Zag扫描的方法？请利用MATLAB的强大功能设计一种最佳方法。

实现Zig-Zag扫描的方法：  
1. **打表法：** 直接手动写出每次扫描对应的数坐标。对于8*8矩阵来说，zig-zag扫描对应的索引值依次是：
1,9,2,3,10,17,25,18,11,4,5,12,19,26,33,41,34,27,20,13,6,7,14,21,28,35,42,49,57,50,43,36,29,22,15,8,16,23,30,37,44,51,58,59,52,45,38,31,24,32,39,46,53,60,61,54,47,40,48,55,62,63,56,64  
将该索引值存入数组，可以直接使用索引值获取扫描后的数组  
2. **扫描法：** 标定一个采样向量，通过程序逻辑实现按顺序扫描。算法如下：
定义当前行`i` ，当前列`j` ，扫描方向`dir` 。扫描方向的取值如图  
每一次循环中，读取当前位置矩阵元素并填入输出数列中，随后根据当前的扫描方向确定下一个位置的坐标，同时根据当前位置决定是否改变扫描方向。
由此，定义zigzag.m函数实现两种采样方法。

```matlab
function y = zigzag(A,mode)
%ZIGZAG zigzag scanning function
% This function returns the zig-zag scan array of the input matrix 
% A, you can select mode from 1(listing mode, only for 8*8 matrix) 
% and 2(automatic coding mode, for all size of matrises).

y = [];
[r,c] = size(A);

if mode==1 % method ONE: Listing
	if r~=8 || c~=8
		error("ZigZag ERROR: Mode ONE input must be 8*8 matrix!");
end
index = [1,9,2,3,10,17,25,18,11,4,5,12,19,26,33,41,34,27,20,13,6,7,14,21,28,35,42,49,57,50,43,36,29,22,15,8,16,23,30,37,44,51,58,59,52,45,38,31,24,32,39,46,53,60,61,54,47,40,48,55,62,63,56,64];
y = A(index);

elseif mode==2 % method TWO: Coding
	i = 1; j = 1; % i row and j column
	dir = 1; % direction: 1-right, 2-downleft, 3-down, 4-upright
	while true
		if i==r && j==c
			y = [y,A(r,c)];
			break;
		end
		y = [y,A(i,j)];
		switch dir
			case 1
				j = j+1;
				if i==1
					dir = 2;
				elseif i==r
					dir = 4;
				end
			case 2
				i = i+1;
				j = j-1;
				if i==r
					dir = 1;
				elseif j==1
					dir = 3;
				end
			case 3
				i = i+1;
				if j==1
					dir = 4;
				elseif j==c
					dir = 2;
				end
			case 4
				i = i-1;
				j = j+1;
				if j==c
					dir = 3;
				elseif i==1
					dir = 1;
				end
		end
	end
end

end
```

随机生成一个8*8矩阵，使用两种方法对矩阵进行zig-zag扫描。

```matlab
%% m2_7.m

clear;
A = rand(8,8);
y1 = zigzag(A,1)
y2 = zigzag(A,2)
```

输出结果如下，可以发现两种方法输出的结果完全相同。  
可以验证当前自定义zig-zag扫描的正确性。

> 8. 对测试图像分块、DCT和量化，将量化后的系数写成矩阵的形式，其中每一列为一个块的DCT系数Zig-Zag扫描后形成的列矢量，第一行为各个块的DC系数。

自定义实现上述预处理的函数`jpeg_preproc.m`

```matlab
function res = jpeg_preproc(pic)
%JPEG_PREPROC preprocess of jpeg encoding
% This function do seperation, DCT transform, quantify, and zig-zag
% scanning of the input picture pic
% Input:
% pic: r*c grayscale matrix (double)
% Output:
% res: 64*(rn*cn) matrix, each column is the zig-zag scan of the DCT coefficient of one block

load data\JpegCoeff.mat QTAB;
% QTAB = QTAB/2;

% seperate the picture into 8*8 blocks
[r,c] = size(pic);
rn = ceil(r/8);
cn = ceil(c/8);
if r<rn*8
	pic(r+1:rn*8,:) = pic(r,:);
end
if c<cn*8
	pic(:,c+1:cn*8) = pic(:,c);
end
pic_block = mat2cell(pic,8*ones(1,rn),8*ones(1,cn)); % blocks

% process
res = zeros(64,rn*cn);
for i=1:rn*cn
	A = cell2mat(pic_block(i)); % preprocess
	C = my_dct2(A-128); % DCT transform
	D = round(C./QTAB); % quantify
	y = zigzag(D,1); % zig-zag scan
	res(:,i) = y;
end

end
```

实现题述要求:

```matlab
%% m2_8.m

clear;
load data\hall.mat hall_gray;
result = jpeg_preproc(hall_gray);
```

结果矩阵（部分）如图。可以发现第一行确实为各个块的DC系数。  

&nbsp;

> 9. 请实现本章介绍的JPEG编码 (不包括写JFIF文件)，输出为DC系数的码流、 AC系数的码流、 图像高度和图像宽度，将这四个变量写入jpegcodes.mat文件。

文件提供的DC和AC编码矩阵为字符形式，不方便使用，我们先通过一个脚本`save_Huffman.m` 将有效信息转化为字符串形式并储存。  
这里我们将DC编码储存为一个 $12\times1$ 字符串数组，第 $i$ 个元素表示 $Category=i-1$ 的Huffman编码；将AC编码储存为一个 $16\times10$ 字符串矩阵，第 $i$ 行 $j$ 列表示 $Run/Size=(i-1)/j$ 的Huffman编码。

```matlab
%% saveHuffman.m

load data\JpegCoeff.mat DCTAB ACTAB;

HDC = strings(12,1);
HAC = strings(160,1);

for i=1:length(HDC)
	for j=2:DCTAB(i,1)+1
		HDC(i) = strcat(HDC(i),num2str(DCTAB(i,j)));
	end
end

for i=1:length(HAC)
	for j=4:ACTAB(i,3)+3
		HAC(i) = strcat(HAC(i),num2str(ACTAB(i,j)));
	end
end
HAC = reshape(HAC,10,16)';

save data\Huffman HDC HAC;
clear HDC HAC DCTAB ACTAB;
```

生成DC码流的函数 `DC_encode.m`  

```matlab
function DC_code = DC_encode(DC)
%DC_ENCODE encoder of DC component array
% This function gets the DC code of the input DC array.
% Input: 
% DC: 1*n row vector 
% Output:
% DC_code: string scalar

load data\Huffman.mat HDC;

len = length(DC);
DC_dif = ([DC(1),DC(1:end-1)-DC(2:end)])'; % differential encoding
DC_cat = zeros(len,1); % category
DC_cat(DC_dif~=0) = floor(log2(abs(DC_dif(DC_dif~=0))))+1;
DC_huf = HDC(DC_cat+1); % Huffman code

% error code
DC_err = strings(len,1);
for i = 1:len
	err = dec2bin(abs(DC_dif(i)));
	if DC_dif(i)<0
		DC_err(i) = string(char(~(err-'0')+'0'));
	else
		DC_err(i) = string(err);
	end
end

DC_codes = strcat(DC_huf,DC_err); % join the code of each DC
DC_code = join(DC_codes,""); % join all DCs

end
```

生成AC码流的函数 `AC_encode.m`  

```matlab
function AC_code = AC_encode(AC)
%AC_ENCODE encoder of AC component matrix
% This function gets the AC code of the input AC matrix.
% Input: 
% AC: 63*n matrix 
% Output:
% AC_code: string scalar

load data\Huffman.mat HAC;
EOB = "1010";
ZRL = "11111111001";

[r,c] = size(AC);
AC_codes = strings(1,c);
for i=1:c
	AC_i = AC(:,i);
	num_0 = 0; % note number of 0 at each point
	for j=1:r
		if AC_i(j)==0
			num_0 = num_0+1;
		else
			% substitute 16*0 with ZRL
			while num_0>=16 
				AC_codes(i) = strcat(AC_codes(i),ZRL);
				num_0 = num_0-16;
			end
			% caculate size and amplitute of data
			size_j = floor(log2(abs(AC_i(j))))+1;
			amp = dec2bin(abs(AC_i(j)));
			if AC_i(j)<0
				amp_j = string(char(~(amp-'0')+'0'));
			else
				amp_j = string(amp);
			end
			AC_codes(i) = strcat(AC_codes(i),HAC(num_0+1,size_j),amp_j);
			num_0 = 0;
		end
	end
	AC_codes(i) = strcat(AC_codes(i),EOB); % end the code with EOB
end
AC_code = join(AC_codes,'');

end
```

代入指导书上的DC和AC信号样例，经检验输出码流完全正确。

综合成JPEG编码的实现函数 `jpeg.m`  

```matlab
function [DC_code,AC_code,height,width] = jpeg(pic)
%JPEG jpeg encoding
% This function returns the jpeg encoding result of the input 
% picture.
% Input:
% pic: r*c grayscale matrix (uint8)
% Output:
% DC_code: DC code stream (string)
% AC_code: AC code stream (string)
% height,width: height and width of input pic

[height,width] = size(pic);
res = jpeg_preproc(double(pic));
DC_code = DC_encode(res(1,:));
AC_code = AC_encode(res(2:end,:));

end
```

获取实验图片的jpeg编码信息并储存。

```matlab
%% m2_9.m

clear;
load data\hall.mat hall_gray;
[DC_code,AC_code,height,width] = jpeg(hall_gray);
save data\jpegcodes DC_code AC_code height width;
```

&nbsp;

> 10. 计算压缩比 (输入文件长度/输出码流长度)，注意转换为相同进制。

代码如下：  

```matlab
%% m2_10.m

clear;
load data\jpegcodes.mat;
size_original = height*width*8
size_compress = strlength(DC_code)+strlength(AC_code)
ratio = size_original/size_compress
```

输出结果为  
压缩比约为6.4，由此可见，jpeg编码方式能节省许多内存空间。

&nbsp;

> 11. 请实现本章介绍的JPEG解码,输入是你生成的jpegcodes.mat文件。分别用客观 (PSNR) 和主观方式评价编解码效果如何。

首先对DC码流进行解码。由于DC码流含有Huffman编码和误差编码，自定义两个函数 `getCategory.m` 和 `getError.m` 读取编码中的 $Category$ 值和误差值。

```matlab
function [category,newcode] = getCategory(code)
%GETCATEGORY Get first category of input Huffman code
% This function returns the first category and the remaining code after removing the first category of the input Huffman code.

load data\Huffman.mat HDC;
for i=1:12
	if startsWith(code,HDC(i))
		category = i-1;
		newcode = eraseBetween(code,1,strlength(HDC(i)));
		return;
	end
end
error("Get Category ERROR: No matching Huffman code!");

end
```

```matlab
function [err,newcode] = getError(code,category)
%GETERROR Get first error value of input error code
% This function returns the first error value and the remaining code after removing the first error value of the input code.

if category==0
	category = 1;
end
newcode = extractAfter(code,category);
err_b = extractBefore(code,category+1);
if extract(err_b,1)=="1"
	err = bin2dec(err_b);
else
	err_b = string(char(~(char(err_b)-'0')+'0'));
	err = -bin2dec(err_b);
end

end
```

基于以上解码函数实现DC码流的解码函数 `DC_decode.m` 

```matlab
function DC = DC_decode(DC_code)
%DC_DECODE decoder of DC code
% This function returns the DC array of the input DC code stream.
% Input:
% DC_code: string scalar
% Output:
% DC: 1*n row vector

DC_err = [];
code = DC_code;
while code ~= ""
	[cat,code] = getCategory(code);
	[err,code] = getError(code,cat);
	DC_err = [DC_err,err];
end
DC = [DC_err(1),DC_err(1)-cumsum(DC_err(2:end))];

end
```

接下来对AC码流进行解码。由于AC码流中含有游程编码和amplitude编码，因此我们需要自定义函数来获取 $Run/Size$ 值和 $Amplitude$ 值。但由于 $Amplitude$ 值的获取方法与DC解码中 $error$ 值的获取方法完全相同，这里复用 `getError.m` 函数并输入 $Size$ 值作为替代，只自定义 `getRunSize.m` 函数。

```matlab
function [run,size,newcode] = getRunSize(code)
%GETRUNSIZE Get first Run/Size value of input Huffman code
% This function returns the first Run/Size value couple of the input Huffman code

load data\Huffman.mat HAC;
if startsWith(code,"1010") % EOB
	run = 0;
	size = 0;
	newcode = eraseBetween(code,1,4);
	return;
elseif startsWith(code,"11111111001") % ZRL
	run = 16;
	size = 0;
	newcode = eraseBetween(code,1,11);
	return;
end
for i=1:16
	for j=1:10
		if startsWith(code,HAC(i,j))
			run = i-1;
			size = j;
			newcode = eraseBetween(code,1,strlength(HAC(i,j)));
			return;
		end
	end
end
error("Get Run/Size ERROR: No matching Huffman code!");

end
```

基于以上解码函数实现AC码流的解码函数 `AC_decode.m` 

```matlab
function AC = AC_decode(AC_code)
%AC_DECODE decoder of AC code
% This function returns the AC array of the input AC code stream.
% Input:
% AC_code: string scalar
% Output:
% AC: 63*n matrix

AC = zeros(63,1);
i = 0; j = 1; % row i and column j
code = AC_code;
while code~=""
	[run,size,code] = getRunSize(code);
	if run==0 && size==0
		i = 0;
		j = j+1;
		continue;
	elseif run==16 && size==0
		i = i+16;
		continue;
	end
	i = i+run+1;
	[num,code] = getError(code,size);
	AC(i,j) = num;
end

end
```

再自定义实现第8问中jpeg预处理过程的逆过程 `dejpeg_postproc.m` 。直接通过将第8问中 `jpeg_preproc.m` 中的处理步骤替换为逆变换并颠倒顺序可得。

```matlab
function pic = dejpeg_postproc(res,height,width)
%DEJPEG_POSTPROC postprocess of jpeg decoding
% This function do inverse zig-zag scan, inverse quantify, inverse DCT transform, and build-up of the input result matrix.
% Input:
% res: 64*(rn*cn) matrix, each column is the zig-zag scan of the DCT coefficient of one block
% height,width: height and width of the output pic
% Output:
% pic: r*c grayscale matrix (double)

load data\JpegCoeff.mat QTAB;
% QTAB = QTAB/2;

rn = ceil(height/8);
cn = ceil(width/8);
pic_block = cell(rn,cn);
for i=1:rn*cn
	y = res(:,i); % extract
	D = izigzag(y); % inverse zig-zag
	C = D.*QTAB; % inverse quantify
	A = my_idct2(C)+128; % inverse DCT
	pic_block(i) = mat2cell(A,8,8); % plug in a cell
end
pic = cell2mat(pic_block); % build-up
pic = pic(1:height,1:width); % cut the extra part

end
```

最后综合成jpeg解码函数 `dejpeg.m` 

```matlab
function pic = dejpeg(DC_code,AC_code,height,width)
%DEJPEG jpeg decoding
% This function returns the jpeg decoding result of the input code
% Input:
% DC_code: DC code stream (string)
% AC_code: AC code stream (string)
% height,width: height and width of output pic
% Output:
% pic: r*c grayscale matrix (uint8)

DC = DC_decode(DC_code);
AC = AC_decode(AC_code);
res = [DC;AC];
pic = uint8(dejpeg_postproc(res,height,width));

end
```

调用 `dejpeg.m` 对保存的jpeg数据进行解码，计算峰值信噪比PSNR，并通过作图直观比较jpeg编解码前后的图片质量。

```matlab
%% m2_11.m

clear;
load data\jpegcodes.mat;
load data\hall.mat hall_gray;
hall_gray_jpeg = dejpeg(DC_code,AC_code,height,width);

%% quantitative comparison
MSE = mean((hall_gray_jpeg-hall_gray).^2,'all');
PSNR = 10*log10(255^2/MSE)

%% plot
figure;
subplot(1,2,1);
imshow(hall_gray);
title("Original");
subplot(1,2,2);
imshow(hall_gray_jpeg);
title("JPEG process");
```

输出峰值信噪比 PSNR 为 34.9090dB，查询维基百科可知，一般的图像压缩 PSNR 值就在 30~40dB 间，可以看出压缩效果较好。  

作图如下。可以发现，JPEG 处理前后的图片，肉眼几乎看不出什么区别。如果仔细观察可以发现，解压后的图像显得更加平缓（例如仔细观察图中大礼堂门口的柱子处）。由此可见，JPEG 的确是一种优秀的图像压缩方式。  

&nbsp;

> 12．将量化步长减小为原来的一半，重做编解码。同标准量化步长的情况比较压缩比和图像质量。

这里我们只需要将使用的 QTAB 矩阵用 QTAB/2 矩阵替换即可。  
（即解除 `jpeg_preproc.m` 和 `dejpeg_postproc.m` 中相应注释）

```matlab
%% m2_12.m

clear;
load data\hall.mat hall_gray;
[DC_code,AC_code,height,width] = jpeg(hall_gray);
hall_gray_jpeg = dejpeg(DC_code,AC_code,height,width);

%% compression ratio
size_original = height*width*8;
size_compress = strlength(DC_code)+strlength(AC_code);
ratio = size_original/size_compress

%% PSNR
MSE = mean((hall_gray_jpeg-hall_gray).^2,'all');
PSNR = 10*log10(255^2/MSE)

%% plot
figure;
subplot(1,2,1);
imshow(hall_gray);
title("Original");
subplot(1,2,2);
imshow(hall_gray_jpeg);
title("JPEG process 2");
```

输出压缩比和 PSNR 

作图如下，同样可以发现，处理前后的图片并没有明显区别。  

对量化步长减小前后的处理效果进行对比，如下表

| QTAB | 压缩比 | PSNR | 主观感受 |
| :-: | :-: | :-: | :-: |
| 原版 | 6.3881 | 34.9090 | 无明显变化 |
| 减半 | 4.3906 | 37.3178 | 无明显变化 |

可以看出，将 QTAB 减半之后，压缩比显著减小，PSNR 略有增加。但由于原版 QTAB 对图像的压缩本就较好，因此对图像给人的主观感受影响不大。

&nbsp;

> 13．看电视时偶尔能看到美丽的雪花图像 (见 `snow.mat` )，请对其编解码。和测试图像的压缩比和图像质量进行比较，并解释比较结果。

这里只需要将 `m2_12.m` 中的 hall_gray 图像改为 snow 图像即可。

```matlab
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
```

输出压缩比和PSNR

作图如下

对JPEG处理大礼堂照片和雪花图像的结果作对比，如下表

| 处理对象 | 压缩比 | PSNR | 主观感受 |
| :-: | :-: | :-: | :-: |
| 大礼堂照片 | 6.3881 | 34.9090 | 无明显变化 |
| 雪花图像 | 3.6439 | 29.5590 | 无明显变化 |

可以发现，对比大礼堂照片，JPEG处理雪花图像的压缩比和PSNR都下降了许多，可见JPEG对雪花图像的处理效果并不好。

原因在于，雪花图像与我们平时接收到的现实图像不同，雪花图像是随机图像，并不像我们平时接收到的图像一样是连续图像，而JPEG是根据人眼对连续亮度的东西较为敏感而设计的，故而对雪花图像的压缩效果不好。

&nbsp;

## 三、信息隐藏

&nbsp;

> 1．实现本章介绍的空域隐藏方法和提取方法。验证其抗 JPEG 编码能力。

构造空域隐藏的函数 `spatialHide.m` 和空域提取的函数 `spatialExtract.m` 。

空域隐藏：将待处理字符串转化为八位二进制编码，再逐位插入到待处理图片像素值的最后一位中。默认以 “00000000” 为结束符。

空域提取：提取图片中所有像素值的最后一位，并按每 8 位为一组进行组合。默认读取到 “00000000” 为止。

```matlab
function pic_new = spatialHide(pic,info)
%SPATIALHIDE Hide the input information in each pixel of input pic
% This function hides the binary form of input info in the last position of each pixel, and returns the new pic.

[r,c] = size(pic);
info = [char(info),char(0)]; % set char(0) as end flag
if length(info)*8>r*c
	error("Spatial Hide ERROR: Information too long!");
end
info_bin = dec2bin(info,8); % binary form of information
info_bin_flatten = char(join(string(info_bin),''))';
info_len = length(info_bin_flatten);
pic_bin = dec2bin(pic,8);
pic_bin(1:info_len,8) = info_bin_flatten;
pic_new = reshape(uint8(bin2dec(pic_bin)),[r,c]);

end
```

```matlab
function info = spatialExtract(pic)
%SPATIALEXTRACT Extract information hidden in input pic
% This function extracts the last position of each pixel and put them together to get the hidden message.

pic_bin = dec2bin(pic);
n = floor(size(pic_bin,1)/8);
info_ext_bin = reshape(pic_bin(1:8*n,8),8,n)';
info_ext = char(bin2dec(info_ext_bin));
info_ext_str = join(string(info_ext),'');
info = extractBefore(info_ext_str,char(0));

end
```

对空域隐藏和提取的效果进行检验，并检验其是否能抵抗 JPEG 编码。

```matlab
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
info_extract_jpeg = spatialExtract(hall_gray_pku_jpeg) % extract info from jpeg

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
```

绘图和输出如下：

从输出和图像中可以得出如下结论：
1. 空域隐藏和提取函数无误，可以准确从图像中提取所隐藏的信息；
2. 空域隐藏对图像的影响可忽略不计，如图所示插入了 “Peking University...” 的图片与原图几乎没有任何区别；
3. 空域隐藏完全不抗 JPEG ，因此经过 JPEG 编解码后输出的提取信息为乱码。原因是 JPEG 是有损压缩，处理后附着的信息会被舍弃。

&nbsp;

> 2. 依次实现本章介绍的三种变换域信息隐藏方法和提取方法，分析嵌密方法的隐蔽性以及嵌密后 JPEG 图像的质量变化和压缩比变化。

构造 DCT 域隐藏的函数 `dctHide.m` 和 DCT 域提取信息的函数 `dctExtract.m` 。

DCT 域隐藏：将待处理字符串转化为八位二进制编码，再以三种方法插入到待处理图片 DCT 矩阵值中。默认以 “00000000” 为结束符。

DCT 域提取：以三种方法提取图片中 DCT 矩阵值的信息，并按每 8 位为一组进行组合。默认读取到 “00000000” 为止。

> （1）同空域方法，用信息位逐一替换掉每一个量化后的 DCT 系数的最低位，再进行熵编码。  
> （2）同方法1，用信息位逐一替换掉若干量化后的DCT系数的最低位，再进行熵编码。注意不是每个 DCT 系数都嵌入了信息。  
> （3）先将待隐藏信息用 1，-1 的序列表示，再逐一将信息位追加在每个块 Zig-Zag 顺序的最后一个非零 DCT 系数之后；如果原本该图像块的最后一个系数就不为零，那就用信息位替换该系数。

```matlab
function [pic_new,DC_code_new,AC_code_new] = dctHide(pic,info,mode)
%DCTHIDE1 Hide info in the DCT matric of pic
% This function replaces the lowest bit of the DCT matrix of pic with info.
% Input:
% pic: r*c grayscale matrix (uint8)
% info: (string)
% mode: select mode (1~3)
% Output:
% pic_new: r*c grayscale matrix (double)
% DC_code_new: new DC code stream (string)
% AC_code_new: new AC code stream (string)

info = [char(info),char(0)]; % set char(0) as end flag
info_bin = dec2bin(info,8); % binary form of information
info_bin_flatten = char(join(string(info_bin),''))';
info_len = length(info_bin_flatten); % flatten the information

[h,w] = size(pic);
res = jpeg_preproc(double(pic)); % get DCT result
[r,c] = size(res);
res_bin = dec2bin(res,8);
switch mode
	% Mode 1: replace the lowest bit of every DCT element
	case 1
		if info_len > size(res_bin,1)
			error("Mode 1: Info too long!");
		end
		res_bin(1:info_len,8) = info_bin_flatten; 
		res_new = reshape(bin2dec(res_bin),[r,c]);
		res_new(res<0) = res_new(res<0)-256;
	% Mode 2: replace the lowest bit of every two DCT elements
	case 2
		if 2*info_len-1 > size(res_bin,1)
			error("Mode 2: Info too long!");
		end
		res_bin(1:2:2*info_len-1,8) = info_bin_flatten;
		res_new = reshape(bin2dec(res_bin),[r,c]);
		res_new(res<0) = res_new(res<0)-256;
	% Mode 3: replace the '0' following the last non-zero element of each DCT block with '1' or '-1'
	case 3
		max_len = size(res,2);
		if info_len > max_len
			error("Mode 3: Info too long!");
		end
		inds = zeros(1,max_len);
		for i=1:max_len
			inds(i) = find(res(:,i),1,"last");
		end
		inds(inds<64) = inds(inds<64)+1; % store index of each block
		ind = inds+64*(0:max_len-1); % store index of the whole block
		info_num_flatten = info_bin_flatten*2-97; % 1,0 -> 1,-1
		res_new = res;
		res_new(ind(1:info_len)) = info_num_flatten;
end % insert in the information

pic_new = dejpeg_postproc(res_new,h,w); % get new pic
DC_code_new = DC_encode(res_new(1,:));
AC_code_new = AC_encode(res_new(2:end,:));

end
```

```matlab
function info = dctExtract(pic,mode)
%DCTEXTRACT Extract info from the DCT matric of pic 
% This function extracts info from the lowest bit of the DCT matrix of pic.
% Input:
% pic: r*c grayscale matrix (double)
% mode: select mode (1~3)
% Output:
% info: (string)

res = jpeg_preproc(pic);
res_bin = dec2bin(res,8);
switch mode
	% Mode 1: extract the lowest bit of every DCT element
	case 1
		n = floor(size(res_bin,1)/8);
		info_ext_bin = reshape(res_bin(1:8*n,8),8,n)';
	% Mode 2: extract the lowest bit of every two DCT elements
	case 2
		n = floor(size(res_bin,1)/16);
		info_ext_bin = reshape(res_bin(1:2:16*n,8),8,n)';
	% Mode 3: extract the last non-zero element of each DCT block
	case 3
		max_len = size(res,2);
		inds = zeros(1,max_len);
		for i=1:max_len
			inds(i) = find(res(:,i),1,"last");
		end
		ind = inds+64*(0:max_len-1); % get the index of each block
		info_num = res(ind); % get the elements
		info_num(info_num~=1&info_num~=-1) = -1; % set others 0
		info_bin = char((info_num+97)/2); % -1,1 -> 0,1
		n = floor(length(info_bin)/8);
		info_ext_bin = reshape(info_bin(1:8*n),8,n)';
end

info_ext = char(bin2dec(info_ext_bin));
info_ext_str = join(string(info_ext),'');
info = extractBefore(info_ext_str,char(0));

end

脚本验证三种隐藏和提取信息函数的正常功能：

```matlab
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
```

输出和作图如下

从结果中我们可以看出：
1. 三种隐藏信息的方法都能够完全正确地提取出对应的信息；
2. 在满额储存信息的情况下，第一、第二种方法储存的信息量要远大于第三种方法；
3. 从第一种方法到第三种方法，压缩率和图像质量依次上升，第三种方法隐藏信息后几乎与隐藏前完全一致，隐蔽性很好。

综上列出以下评价表

|方法种类 | 信息量 | 压缩比 | 图像质量（PSNR） | 隐蔽性 |
| :-: | :-: | :-: | :-: | :-: |
| 1 | 2500+字符（大） | 2.9172（低） | 28.3079（差） | 差 |
| 2 | 1300+字符（较大） | 3.5468（较低） | 28.7469（差） | 差 |
| 3 | 20+字符（小） | 6.2806（较高） | 34.1054（较好） | 较好 |

&nbsp;

> 3. （选做）请设计实现新的隐藏算法并分析其优缺点。

不会，以后再做。

&nbsp;

## 四、人脸检测

&nbsp;

> 1. 所给资料 Faces 目录下包含从网图中截取的 28 张人脸，试以其作为样本训练人脸标准 v。

> （1）样本人脸大小不一，是否需要首先将图像调整为相同大小？

不需要，因为检测方式是通过区域内各种颜色出现的频率进行判断，并不需要图像整体大小相同。

> （2）假设L分别取3，4，5，所得三个v之间有何关系？

当 $L$ 取 3/4/5 时，我们取每个像素点的八位 uint8 像素值的前 3/4/5 位作为特征计算向量 $u$ ，即 $u(L=3)$ 的每一个元素是 $u(L=4)$ 的对应两个元素（第 4 位取 0/1 ）之和，以此类推。所得 $v$ 也有相同的关系。

&nbsp;

> 2. 设计一种从任意大小的图片中检测任意多张人脸的算法并编程实现（输出图像在判定为人脸的位置加上红色的方框）。随意选取一张多人照片（比如支部活动或者足球比赛），对程序进行测试。尝试 $L$ 分别取不同的值，评价检测结果有何区别。

1. 构造函数 `getFeature.m` ，实现从给定图像中获取颜色频率向量 $u$ 。  
这里我们通过把 `pic` 每个像素值都除以 $2^L$ 再取整，将结果直接作为频率矩阵的下标进行计数，来获取 $2^{3\times L}$ 种颜色的出现频率。

```matlab
function u = getFeature(pic,L)
%GETFEATURE Get the feature matrix u of pic and parameter L
% This function returns the color frequency matrix u of input pic
% Input:
% pic: h*w*3 RGB picture
% L: precision parameter of color recognition (3~8)
% Output:
% u: (2^L)*(2^L)*(2^L) feature matrix, u(r,g,b) represents the frequency of color (r,g,b) in the input pic.

% preprocess
[h,w,~] = size(pic);
pic_L = floor(double(pic)/2^(8-L));
pic_flatten = reshape(pic_L,h*w,3);
u_index = pic_flatten*[1;2^L;2^(2*L)]+1;

% count
u = zeros(2^L,2^L,2^L);
for i=1:h*w
	u(u_index(i)) = u(u_index(i))+1;
end
u = u/sum(u,'all');
end
```

2. 构造函数 `trainFeature.m` ，导入所给的 33 张人脸图片，依次调用上述函数提取出特征矩阵 $u$ ，并取平均得到标准特征矩阵 $v$ 。

```matlab
function v = trainFeature(pics,L)
%TRAINFEATURE Train on the given face pics to get standard v
% This function reads and extracts u from all pic in pics, and take an average to get standard v.

n = length(pics);
us = zeros(2^L,2^L,2^L,n);
for i=1:n
	pic = cell2mat(pics(i));
	us(:,:,:,i) = getFeature(pic,L);
end
v = mean(us,4);

end
```

我们通过一个脚本 `saveFaceStandard.m` 来获取并存储 $L=3\sim 8$ 对应的矩阵 $v$ 。

```matlab
%% saveFaceStandard.m

faces = cell(33,1);
for i=1:33
	face = imread("data\Faces\"+string(i)+".bmp");
	faces(i) = mat2cell(face,size(face,1),size(face,2),3);
end
vs = cell(1,8);
for i=3:8
	vs(i) = mat2cell(trainFeature(faces,i),2^i,2^i,2^i);
end
save data\facefeature vs;
```

3. 构造函数 `testFeature.m` ，检测输入图片中符合要求的区域。  
这里我们依照 jpeg 编码中图像分块的相同手段，使用元胞 cell 矩阵对输入图像进行分块，然后对每个小块调用 `getFeature` 函数获取特征矩阵 $u$ 并计算误差 $err$ 。若误差小于阈值，则将结果矩阵对应区域标记为 1，否则标记为 0。  
获得初步结果矩阵后，查看输出图像发现绘制的方框过于细碎杂乱，于是加入结果矩阵的处理函数，使得输出图像的方框标记均为矩形，并去除孤立小方框（大多是由于对环境颜色的误判产生的）。

```matlab
function res = testFeature(pic,bsize,L,eps)
%TESTFEATURE Test which part of pic meets the requirements
% This function splits the input pic into blocks, tests each block, and returns a 0-1 matrix which marks the regions.
% Input:
% pic: h*w*3 color picture
% bsize: square block size
% L: precision parameter of color recognition (3~8)
% eps: maximum error
% Output:
% res: h*w logical matrix

% split pic into blocks
[height,width,~] = size(pic);
rn = floor(height/bsize);
cn = floor(width/bsize);
if rn*bsize<height
	rns = [bsize*ones(1,rn),height-bsize*rn];
elseif rn*bsize==height
	rns = bsize*ones(1,rn);
end
if cn*bsize<width
	cns = [bsize*ones(1,cn),width-bsize*cn];
elseif cn*bsize==width
	cns = bsize*ones(1,cn);
end
pic_block = mat2cell(pic,rns,cns,3);
res_block = mat2cell(zeros(height,width),rns,cns);

% calculate vector u and error for each block
load data\facefeature.mat vs;
v = cell2mat(vs(L));
[r,c] = size(pic_block);
res_block_val = zeros(r,c);
for i=1:r*c
	pblock = cell2mat(pic_block(i));
	rblock = cell2mat(res_block(i));
	u = getFeature(pblock,L);
	err = 1-sum(sqrt(u.*v),'all');
	if err<eps % if satisfied, turn res to 1
		res_block_val(i) = 1;
		rblock = ones(size(rblock));
		res_block(i) = mat2cell(rblock,size(rblock,1),size(rblock,2));
	end
end

% process and transform the result blocks
done = 0;
while ~done
	[res_block,res_block_val,done] = res_process(res_block,res_block_val);
end
res = cell2mat(res_block);

end
```

这里调用了 `res_process` 函数对结果矩阵进行处理，处理原理如下：  
① 去除孤立方框：  
采用判断：若某值为 1 的块上下（或左右）的块均为 0，那么判定该块为“孤立块”，手动将该块的值赋为 0。  
② 将边框均转换为矩形：  
采用判断：若某值为 1 的块上下左右的块满足如下图所示的情况，则将对应块的值手动赋为 1。  

设置一个标记处理是否结束的变量，将上述操作循环若干遍，直到某一遍循环时未进行 ①② 中的任何操作，则认定为完成，跳出循环，得到处理后的矩阵。

```matlab
function [res_new,vals_new,done] = res_process(res_block,vals)
%RES_PROCESS Process the result matrix 
% This function process the result logical matrix so that the result boxes can be more beautiful.
% Input:
% res_block: result matrix (cell matrix)
% vals: value of result cells (double matrix)
% Output:
% res_new: result matrix after processing (cell matrix)
% vals_new: value of result cells after processing (double matrix)
% done: mark whether the process is finished


% subfunction
function block_new = set_block_value(val,block)
mat = cell2mat(block);
mat = val*ones(size(mat));
block_new = mat2cell(mat,size(mat,1),size(mat,2));
end


res_new = res_block;
vals_new = vals;
done = 1;
[r,c] = size(res_block);
for i=1:r
	for j=1:c
		if vals_new(i,j)==1
			% discard the isolated blocks
			if ((i>1 && vals_new(i-1,j)==0) && (i<r && vals_new(i+1,j)==0)) || ((j>1 && vals_new(i,j-1)==0) && (j<c && vals_new(i,j+1)==0))
				vals_new(i,j) = 0;
				res_new(i,j) = set_block_value(0,res_new(i,j));
				done = 0;
			end
			% fill the stair blocks (4 cases)
			% case 1
			if (i>1 && vals_new(i-1,j)==1) && (j>1 && vals_new(i,j-1)==1) && (vals_new(i-1,j-1)==0)
				vals_new(i-1,j-1) = 1;
				res_new(i-1,j-1) = set_block_value(1,res_new(i-1,j-1));
				done = 0;
			end
			% case 2
			if (i>1 && vals_new(i-1,j)==1) && (j<c && vals_new(i,j+1)==1) && (vals_new(i-1,j+1)==0)
				vals_new(i-1,j+1) = 1;
				res_new(i-1,j+1) = set_block_value(1,res_new(i-1,j+1));
				done = 0;
			end
			% case 3
			if (i<r && vals_new(i+1,j)==1) && (j>1 && vals_new(i,j-1)==1) && (vals_new(i+1,j-1)==0)
				vals_new(i+1,j-1) = 1;
				res_new(i+1,j-1) = set_block_value(1,res_new(i+1,j-1));
				done = 0;
			end
			% case 4
			if (i<r && vals_new(i+1,j)==1) && (j<c && vals_new(i,j+1)==1) && (vals_new(i+1,j+1)==0)
				vals_new(i+1,j+1) = 1;
				res_new(i+1,j+1) = set_block_value(1,res_new(i+1,j+1));
				done = 0;
			end
		end
	end
end

end
```

4. 验证上述函数的功能并作图。  
这里我们使用 `visboundaries` 函数对生成的 logical 矩阵 res 进行边框绘制。

```matlab
%% m4_2.m

clear;
img = imread("data\image.jpg");
res = testFeature(img,64,5,0.8);

%% plot
f = figure;
imshow(img);
hold on;
B = bwboundaries(res); % create boundary
visboundaries(B); % draw boundary
saveas(f,"data\image_proc.jpg");
```

这里我选取了我们学生赣文化交流协会三次不同活动的合影，合影人数从多到少，人脸大小从大到小，依次测试人脸检测程序的检测效果和方框处理程序的处理效果。

通过以上三张图片的处理可以得出结论：以肤色识别为核心的人脸识别模型更适用于满足以下条件的图片：  
①人脸数量相对较少  
②人脸大小相对较大  
③人脸之间间隔相对较远  
④除人脸外其他相似颜色的物体（如皮肤、环境颜色等）较少  
⑤光线较好，不过亮也不过暗  



