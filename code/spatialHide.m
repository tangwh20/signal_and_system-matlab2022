function pic_new = spatialHide(pic,info)
%SPATIALHIDE Hide the input information in each pixel of the input pic
%   This function hides the binary form of input info in the last position
%   of each pixel, and returns the new pic.

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