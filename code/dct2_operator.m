function D = dct2_operator(N)
%DCT2_OPERATOR operator matrix D of 2-dimensional DCT transform
%   This function returns N*N matrix D based on the input number N

v1 = 1:2:2*N-1;
v2 = (0:N-1)';
D = sqrt(2/N)*cos(kron(v1,v2)*pi/2/N);
D(1,:) = D(1,:)/sqrt(2);

end