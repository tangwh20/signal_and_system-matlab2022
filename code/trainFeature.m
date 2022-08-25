function v = trainFeature(pics,L)
%TRAINFEATURE Train our model on the given face pics to get standard v
%   This function reads and extracts u from all pic in pics, and take an
%   average to get standard v.

n = length(pics);
us = zeros(2^L,2^L,2^L,n);
for i=1:n
    pic = cell2mat(pics(i));
    us(:,:,:,i) = getFeature(pic,L);
end
v = mean(us,4);

end