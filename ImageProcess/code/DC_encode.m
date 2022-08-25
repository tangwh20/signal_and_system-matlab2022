function DC_code = DC_encode(DC)
%DC_ENCODE encoder of DC component array
%   This function gets the DC code of the input DC array.
%   Input: 
%   DC: 1*n row vector 
%   Output:
%   DC_code: string scalar

load data\Huffman.mat HDC;

len = length(DC);
DC_dif = ([DC(1),DC(1:end-1)-DC(2:end)])'; % differential encoding
DC_cat = zeros(len,1); % category
DC_cat(DC_dif~=0) = floor(log2(abs(DC_dif(DC_dif~=0))))+1;
DC_huf = HDC(DC_cat+1); % Huffman code

% error code
DC_err = strings(len,1);
for i = 1:len
    err = dec2bin(abs(DC_dif(i)));
    if DC_dif(i)==0
        DC_err(i) = "";
    elseif DC_dif(i)<0
        DC_err(i) = string(char(~(err-'0')+'0'));
    else
        DC_err(i) = string(err);
    end
end

DC_codes = strcat(DC_huf,DC_err); % join the code of each DC
DC_code = join(DC_codes,""); % join all DCs

end