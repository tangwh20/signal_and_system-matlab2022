function info = spatialExtract(pic)
%SPATIALEXTRACT Extract information hidden in input pic
%   This function extracts the last position of each pixel and put them
%   together to get the hidden message.

pic_bin = dec2bin(pic);
n = floor(size(pic_bin,1)/8);
info_ext_bin = reshape(pic_bin(1:8*n,8),8,n)';
info_ext = char(bin2dec(info_ext_bin));
info_ext_str = join(string(info_ext),'');
info = extractBefore(info_ext_str,char(0));

end