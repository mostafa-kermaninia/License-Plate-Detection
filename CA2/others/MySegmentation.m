clc
close all;
clear;
% SELECTING THE TEST DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[file,path]=uigetfile({'*.jpg;*.png;*.jpeg'},'Select your image');
picture= imread(fullfile(path, file));

picture=imresize(picture,[300 500]);
%RGB2GRAY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
picture=rgb2gray(picture);
% THRESHOLDIG and CONVERSION TO A BINARY IMAGE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
threshold = graythresh(picture);
picture = imbinarize(picture,threshold);
picture=~picture;
%%%%%%%

[row,col]=find(picture==1);
POINTS=[row';col'];
POINTS_NUM=size(POINTS,2);

initpoint=POINTS(:,1);
POINTS(:,1)=[];
POINTS_NUM=POINTS_NUM-1;
CurrectObject=[initpoint];
t=1;
while POINTS_NUM>0
    
    [POINTS,newPoints]=close_points(initpoint,POINTS);
    newPoints_len=size(newPoints,2);
    CurrectObject=[CurrectObject newPoints];
   
    while newPoints_len>0
%         for i=1:newPoints_len
            initpoint=newPoints(:,1);
            newPoints(:,1)=[];
            [POINTS,newPoints2]=close_points(initpoint,POINTS);            
            CurrectObject=[CurrectObject newPoints2];
            newPoints=[newPoints newPoints2];
            newPoints_len=size(newPoints,2);
%         end
    end
   
    FINALOBJECT{t}=CurrectObject;
    t=t+1;
    POINTS_NUM=size(POINTS,2);
    if POINTS_NUM>0
        initpoint=POINTS(:,1);
        CurrectObject=initpoint;
    end
 
end
    disp('');

X=zeros(size(picture));


for i=1:size(FINALOBJECT{14},2)
    X(FINALOBJECT{14}(1,i),FINALOBJECT{14}(2,i))=1;
end
imshow(X)






%%%
pictureMATLAB = bwareaopen(picture,500); % removes all connected components (objects) that have fewer than 500 pixels from the binary image
