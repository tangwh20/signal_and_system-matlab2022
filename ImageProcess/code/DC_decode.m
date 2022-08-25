function DC = DC_decode(DC_code)
%DC_DECODE decoder of DC code
%   This function returns the DC array of the input DC code stream.
%   Input:
%   DC_code: string scalar
%   Output:
%   DC: 1*n row vector

DC_err = [];
code = DC_code;
while code~=""
    [cat,code] = getCategory(code);
    [err,code] = getError(code,cat);
    DC_err = [DC_err,err];
end
DC = [DC_err(1),DC_err(1)-cumsum(DC_err(2:end))];

end