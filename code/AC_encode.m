function AC_code = AC_encode(AC)
%AC_ENCODE encoder of AC component matrix
%   This function gets the AC code of the input AC matrix.
%   Input: 
%   AC: 63*n matrix 
%   Output:
%   AC_code: string scalar

load data\Huffman.mat HAC;
EOB = "1010";
ZRL = "11111111001";

[r,c] = size(AC);
AC_codes = strings(1,c);
for i=1:c
    AC_i = AC(:,i);
    num_0 = 0; % note number of 0 at each point
    for j=1:r
        if AC_i(j)==0
            num_0 = num_0+1;
        else
            % substitute 16*0 with ZRL
            while num_0>=16 
                AC_codes(i) = strcat(AC_codes(i),ZRL);
                num_0 = num_0-16;
            end
            % caculate size and amplitute of data
            size_j = floor(log2(abs(AC_i(j))))+1;
            amp = dec2bin(abs(AC_i(j)));
            if AC_i(j)==0
                amp_j = "";
            elseif AC_i(j)<0
                amp_j = string(char(~(amp-'0')+'0'));
            else
                amp_j = string(amp);
            end
            % join them together
            AC_codes(i) = strcat(AC_codes(i),HAC(num_0+1,size_j),amp_j);
            num_0 = 0;
        end
    end
    AC_codes(i) = strcat(AC_codes(i),EOB); % end the code with EOB
end
AC_code = join(AC_codes,'');
end