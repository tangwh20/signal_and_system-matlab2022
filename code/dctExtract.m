function info = dctExtract(pic,mode)
%DCTEXTRACT Extract info from the DCT matric of pic 
%   This function extracts info from the lowest bit of the DCT matrix of pic.
%   Input:
%   pic: r*c grayscale matrix (double)
%   mode: select mode (1~3)
%   Output:
%   info: (string)

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