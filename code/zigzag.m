function y = zigzag(A,mode)
%ZIGZAG zigzag scanning function
%   This function returns the zig-zag scan array of the input matrix A, you
%   can select mode from 1(listing mode, only for 8*8 matrix) and
%   2(automatic coding mode, for all size of matrises).

y = [];
[r,c] = size(A);
if mode==1 % method ONE: Listing
    if r~=8 || c~=8
        error("ZigZag ERROR: Mode ONE input must be 8*8 matrix!");
    end
    index = [1,9,2,3,10,17,25,18,11,4,5,12,19,26,33,41,34,27,20,13,6,7,...
        14,21,28,35,42,49,57,50,43,36,29,22,15,8,16,23,30,37,44,51,58,...
        59,52,45,38,31,24,32,39,46,53,60,61,54,47,40,48,55,62,63,56,64];
    y = A(index);

elseif mode==2 % method TWO: Coding
    i = 1; j = 1; % i row and j column
    dir = 1; % direction: 1-right, 2-downleft, 3-down, 4-upright
    while true
        if i==r && j==c
            y = [y,A(r,c)];
            break;
        end
        y = [y,A(i,j)];
        switch dir
            case 1
                j = j+1;
                if i==1
                    dir = 2;
                elseif i==r
                    dir = 4;
                end
            case 2
                i = i+1;
                j = j-1;
                if i==r
                    dir = 1;
                elseif j==1
                    dir = 3;
                end
            case 3
                i = i+1;
                if j==1
                    dir = 4;
                elseif j==c
                    dir = 2;
                end
            case 4
                i = i-1;
                j = j+1;
                if j==c
                    dir = 3;
                elseif i==1
                    dir = 1;
                end
        end
    end
end

end