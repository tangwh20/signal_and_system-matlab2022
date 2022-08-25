function res = testFeature(pic,bsize,L,eps)
%TESTFEATURE Test which part of pic meets the requirements
%   This function splits the input pic into blocks, tests each block, and
%   returns a 0-1 matrix which marks the regions.
%   Input:
%   pic: h*w*3 color picture
%   bsize: square block size
%   L: precision parameter of color recognition (3~8)
%   eps: maximum error
%   Output:
%   res: h*w logical matrix

% split pic into blocks
[height,width,~] = size(pic);
rn = floor(height/bsize);
cn = floor(width/bsize);
if rn*bsize<height
    rns = [bsize*ones(1,rn),height-bsize*rn];
elseif rn*bsize==height
    rns = bsize*ones(1,rn);
end
if cn*bsize<width
    cns = [bsize*ones(1,cn),width-bsize*cn];
elseif cn*bsize==width
    cns = bsize*ones(1,cn);
end
pic_block = mat2cell(pic,rns,cns,3);
res_block = mat2cell(zeros(height,width),rns,cns);

% calculate vector u and error for each block
load data\facefeature.mat vs;
v = cell2mat(vs(L));
[r,c] = size(pic_block);
res_block_val = zeros(r,c);
for i=1:r*c
    pblock = cell2mat(pic_block(i));
    rblock = cell2mat(res_block(i));
    u = getFeature(pblock,L);
    err = 1-sum(sqrt(u.*v),'all');
    if err<eps % if satisfied, turn res to 1
        res_block_val(i) = 1;
        rblock = ones(size(rblock));
        res_block(i) = mat2cell(rblock,size(rblock,1),size(rblock,2));
    end
end

% process and transform the blocks
done = 0;
while ~done
    [res_block,res_block_val,done] = res_process(res_block,res_block_val);
end
res = cell2mat(res_block);

end