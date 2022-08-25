function [pic_new,DC_code_new,AC_code_new] = dctHide(pic,info,mode)
%DCTHIDE1 Hide info in the DCT matric of pic
%   This function replaces the lowest bit of the DCT matrix of pic with info.
%   Input:
%   pic: r*c grayscale matrix (uint8)
%   info: (string)
%   mode: select mode (1~3)
%   Output:
%   pic_new: r*c grayscale matrix (double)
%   DC_code_new: new DC code stream (string)
%   AC_code_new: new AC code stream (string)

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
    % Mode 3: replace the '0' following the last non-zero element of each
    % DCT block with '1' or '-1'
    case 3
        max_len = size(res,2);
        if info_len > max_len
            error("Mode 3: Info too long!");
        end
        inds = zeros(1,max_len);
        for i=1:max_len
            inds(i) = find(res(:,i),1,"last");
        end
        inds(inds<64) = inds(inds<64)+1; % store the index of each block
        ind = inds+64*(0:max_len-1); % store the index of whole block
        info_num_flatten = info_bin_flatten*2-97; % transform to 1 and -1
        res_new = res;
        res_new(ind(1:info_len)) = info_num_flatten;
end % insert in the information

pic_new = dejpeg_postproc(res_new,h,w); % get new pic
DC_code_new = DC_encode(res_new(1,:));
AC_code_new = AC_encode(res_new(2:end,:));

end