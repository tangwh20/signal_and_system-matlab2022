function [run,size,newcode] = getRunSize(code)
%GETRUNSIZE Get first Run/Size value of input Huffman code
%   This function returns the first Run/Size value couple of the input 
%   Huffman code

load data\Huffman.mat HAC;
if startsWith(code,"1010")
    run = 0;
    size = 0;
    newcode = eraseBetween(code,1,4);
    return;
elseif startsWith(code,"11111111001")
    run = 16;
    size = 0;
    newcode = eraseBetween(code,1,11);
    return;
end
for i=1:16
    for j=1:10
        if startsWith(code,HAC(i,j))
            run = i-1;
            size = j;
            newcode = eraseBetween(code,1,strlength(HAC(i,j)));
            return;
        end
    end
end
error("Get Run/Size ERROR: No matching Huffman code!");

end