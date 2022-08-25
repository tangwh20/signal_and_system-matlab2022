function u = getFeature(pic,L)
%GETFEATURE Get the feature matrix u of pic and parameter L
%   This function returns the color frequency matrix u of input pic
%   Input:
%   pic: h*w*3 RGB picture
%   L: precision parameter of color recognition (3~8)
%   Output:
%   u: (2^L)*(2^L)*(2^L) feature matrix, u(r,g,b) represents the frequency
%   of color (r,g,b) in the input pic.

% preprocess
[h,w,~] = size(pic);
pic_L = floor(double(pic)/2^(8-L));
pic_flatten = reshape(pic_L,h*w,3);
u_index = pic_flatten*[1;2^L;2^(2*L)]+1;
% count
u = zeros(2^L,2^L,2^L);
for i=1:h*w
    u(u_index(i)) = u(u_index(i))+1;
end
u = u/sum(u,'all');
end