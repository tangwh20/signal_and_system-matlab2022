function C = my_dct2(A)
%MY_DCT2 2-dimensional DCT transform
%   This function returns the 2-dimensional DCT transform of the input
%   matrix A

[m,n] = size(A);
Dm = dct2_operator(m);
Dn = dct2_operator(n);
C = Dm*A*Dn';

end