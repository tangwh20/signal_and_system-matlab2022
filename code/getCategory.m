function [category,newcode] = getCategory(code)
%GETCATEGORY Get first category of input Huffman code
%   This function returns the first category and the remaining code after
%   removing the first category of the input Huffman code.

load data\Huffman.mat HDC;
for i=1:12
    if startsWith(code,HDC(i))
        category = i-1;
        newcode = eraseBetween(code,1,strlength(HDC(i)));
        return;
    end
end
error("Get Category ERROR: No matching Huffman code!");

end