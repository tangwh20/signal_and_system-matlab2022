function [err,newcode] = getError(code,category)
%GETERROR Get first error value of input error code
%   This function returns the first error value and the remaining code
%   after removing the first error value of the input code.

newcode = extractAfter(code,category);
err_b = extractBefore(code,category+1);
if err_b==""
    err = 0;
elseif extract(err_b,1)=="1"
    err = bin2dec(err_b);
else
    err_b = string(char(~(char(err_b)-'0')+'0'));
    err = -bin2dec(err_b);
end

end