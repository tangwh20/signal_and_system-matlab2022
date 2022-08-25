function res = jpeg_preproc(pic)
%JPEG_PREPROC preprocess of jpeg encoding
%   This function do seperation, DCT transform, quantify, and zig-zag
%   scanning of the input picture pic
%   Input:
%   pic: r*c grayscale matrix (double)
%   Output:
%   res: 64*(rn*cn) matrix, each column is the zig-zag scan of the DCT
%   coefficient of one block

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