function pic = dejpeg(DC_code,AC_code,height,width)
%DEJPEG jpeg decoding
%   This function returns the jpeg decoding result of the input code
%   Input:
%   DC_code: DC code stream (string)
%   AC_code: AC code stream (string)
%   height,width: height and width of output pic
%   Output:
%   pic: r*c grayscale matrix (uint8)

DC = DC_decode(DC_code);
AC = AC_decode(AC_code);
res = [DC;AC];
pic = uint8(dejpeg_postproc(res,height,width));

end