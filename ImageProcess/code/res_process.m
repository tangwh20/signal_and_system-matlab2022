function [res_new,vals_new,done] = res_process(res_block,vals)
%RES_PROCESS Process the result matrix 
%   This function process the result logical matrix so that the result
%   boxes can be more beautiful
%   Input:
%   res_block: result matrix (cell matrix)
%   vals: value of result cells (double matrix)
%   Output:
%   res_new: result matrix after processing (cell matrix)
%   vals_new: value of result cells after processing (double matrix)
%   done: mark whether the process is finished

% subfunction
function block_new = set_block_value(val,block)
    mat = cell2mat(block);
    mat = val*ones(size(mat));
    block_new = mat2cell(mat,size(mat,1),size(mat,2));
end

res_new = res_block;
vals_new = vals;
done = 1;
[r,c] = size(res_block);
for i=1:r
    for j=1:c
        if vals_new(i,j)==1
            % discard the isolated blocks
            if ((i>1 && vals_new(i-1,j)==0) && (i<r && vals_new(i+1,j)==0)) ...
                    || ((j>1 && vals_new(i,j-1)==0) && (j<c && vals_new(i,j+1)==0))
                vals_new(i,j) = 0;
                res_new(i,j) = set_block_value(0,res_new(i,j));
                done = 0;
            end
            % fill the stair blocks (4 circumstances)
            if (i>1 && vals_new(i-1,j)==1) && (j>1 && vals_new(i,j-1)==1) ...
                    && vals_new(i-1,j-1)==0
                vals_new(i-1,j-1) = 1;
                res_new(i-1,j-1) = set_block_value(1,res_new(i-1,j-1));
                done = 0;
            end
            if (i>1 && vals_new(i-1,j)==1) && (j<c && vals_new(i,j+1)==1) ...
                    && vals_new(i-1,j+1)==0
                vals_new(i-1,j+1) = 1;
                res_new(i-1,j+1) = set_block_value(1,res_new(i-1,j+1));
                done = 0;
            end
            if (i<r && vals_new(i+1,j)==1) && (j>1 && vals_new(i,j-1)==1) ...
                    && vals_new(i+1,j-1)==0
                vals_new(i+1,j-1) = 1;
                res_new(i+1,j-1) = set_block_value(1,res_new(i+1,j-1));
                done = 0;
            end
            if (i<r && vals_new(i+1,j)==1) && (j<c && vals_new(i,j+1)==1) ...
                    && vals_new(i+1,j+1)==0
                vals_new(i+1,j+1) = 1;
                res_new(i+1,j+1) = set_block_value(1,res_new(i+1,j+1));
                done = 0;
            end
        end
    end
end

end