function pic = dejpeg_postproc(res,height,width)
%DEJPEG_POSTPROC postprocess of jpeg decoding
%   This function do inverse zig-zag scan, inverse quantify, inverse DCT
%   transform, and build-up of the input intermediate result matrix.
%   Input:
%   res: 64*(rn*cn) matrix, each column is the zig-zag scan of the DCT
%   coefficient of one block
%   height,width: height and width of the output pic
%   Output:
%   pic: r*c grayscale matrix (double)

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