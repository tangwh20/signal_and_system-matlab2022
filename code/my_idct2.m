function A = my_idct2(C)
%MY_IDCT2 2-dimensional inverse DCT transform
%   This function returns the 2-dimensional inverse DCT transform of the 
%   input matrix C

[m,n] = size(C);
Dm = dct2_operator(m);
Dn = dct2_operator(n);
A = Dm'*C*Dn;
end