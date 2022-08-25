function AC = AC_decode(AC_code)
%AC_DECODE decoder of AC code
%   This function returns the AC array of the input AC code stream.
%   Input:
%   AC_code: string scalar
%   Output:
%   AC: 63*n matrix

AC = zeros(63,1);
i = 0; j = 1; % row i and column j
code = AC_code;
while code~=""
    [run,size,code] = getRunSize(code);
    if run==0 && size==0
        i = 0;
        j = j+1;
        continue;
    elseif run==16 && size==0
        i = i+16;
        continue;
    end
    i = i+run+1;
    [num,code] = getError(code,size);
    AC(i,j) = num;
end

end