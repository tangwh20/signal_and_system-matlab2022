%% saveHuffman.m

load data\JpegCoeff.mat DCTAB ACTAB;

HDC = strings(12,1);
HAC = strings(160,1);

for i=1:length(HDC)
    for j=2:DCTAB(i,1)+1
        HDC(i) = strcat(HDC(i),num2str(DCTAB(i,j)));
    end
end

for i=1:length(HAC)
    for j=4:ACTAB(i,3)+3
        HAC(i) = strcat(HAC(i),num2str(ACTAB(i,j)));
    end
end
HAC = reshape(HAC,10,16)';

save data\Huffman HDC HAC;
clear HDC HAC DCTAB ACTAB;