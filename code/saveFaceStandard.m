%% saveFaceStandard

faces = cell(33,1);
for i=1:33
    face = imread("data\Faces\"+string(i)+".bmp");
    faces(i) = mat2cell(face,size(face,1),size(face,2),size(face,3));
end
v = cell(1,8);
for i=3:8
    v(i) = mat2cell(trainFeature(faces,i),2^i,2^i,2^i);
end
save data\facefeature v;