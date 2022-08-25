function [DC_code,AC_code,height,width] = jpeg(pic)
%JPEG jpeg encoding
%   This function returns the jpeg encoding result of the input picture
%   Input:
%   pic: r*c grayscale matrix (uint8)
%   Output:
%   DC_code: DC code stream (string)
%   AC_code: AC code stream (string)
%   height,width: height and width of input pic

[height,width] = size(pic);
res = jpeg_preproc(double(pic));
DC_code = DC_encode(res(1,:));
AC_code = AC_encode(res(2:end,:));

end