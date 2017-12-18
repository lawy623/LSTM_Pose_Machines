%%Training Data Preprocessing. Output: Image , Label Heatmap, Centeral Heatmap
function [ image, labelmap, centerMap ] = transformation_PENN( trainBatch, boxsize, stride, np, seqTrain )
%Transformation Parameters (scale/resize, rotate, crop, flip, addLabels)
%Scale
scale_control = 1;        %determine how random is the resize. Default as 1.(Always do).
min_scale = 0.8;
max_scale = 1.4;
scale_rate = min_scale + (max_scale-min_scale)*rand(1);         %random scale between min~max. All frames should be the same.
%Rotate
rotate_control = 1;       %determine how random is the rotation. Default as 1.(Always do).
max_degree = 40;
rotate_degree = -max_degree + 2*(max_degree)*rand(1);           %random rotation between -max~max degree
%Flip
flip_control = 0.5;       %determine how random is the rotation. Default as 0.5.(Do 50% times).
%Gaussian Map
sigma_center = 21;
sigma = 7;

order_to_pretrain = [1 14 2 4 6 3 5 7 8 10 12 9 11 13];

% Whether to do the transformation or not
% Put it here to make the transformation for whole sequence consistent
if rand(1) > scale_control      %Do the resize operation at frequency (100*scale_control)%
   scale_rate=1;
end
if rand(1) > rotate_control     %Do the rotate operation at frequency (100*rotate_control)%
    rotate_degree=0;
end
if rand(1) < flip_control       %Do the flip operation at frequency (100*flip_control)%
    do_flip=1;
else
    do_flip=0;
end

%Starting Index for the first training frames in sequences of batch
startInd = randi(trainBatch.nframes - seqTrain +1 );

%do not get trainBatch.dim since sometime they do not match dimension of frame..
dim = [size(trainBatch.frame{1}.image,1) size(trainBatch.frame{1}.image,2)];

image = uint8(zeros([dim(1),dim(2),3,seqTrain]));
label = zeros([3,np,seqTrain]);
bbox = zeros([seqTrain,4]);

%Read Data from TrainBatch struct.
    for i=1:seqTrain
        image(:,:,:,i) = trainBatch.frame{startInd+i-1}.image;
        label(1,:,i)= trainBatch.frame{startInd+i-1}.label.x;
        label(2,:,i)= trainBatch.frame{startInd+i-1}.label.y;
        label(3,:,i)= double(trainBatch.frame{startInd+i-1}.label.visibility);
        bbox(i,:) = trainBatch.frame{startInd+i-1}.bbox;
    end

    %create label for neck to keep consistence of model. But it will be
    %ignored during testing. We intropolate the pos by head(1) and shoulders(2&3).
    label(1,np+1,:) = 0.5*label(1,1,:) + 0.25*(label(1,2,:)+label(1,3,:));
    label(2,np+1,:) = 0.2*label(2,1,:) + 0.4*(label(2,2,:)+label(2,3,:));
    label(3,np+1,:) = floor( (label(3,1,:)+label(3,2,:)+label(3,3,:))/3 );

    %make the joints not in the figure vis=-1(Do not produce label)
    for i=1:seqTrain
        for part = 1:size(label,2)
            if isNotOnPlane(label(1,part,i),label(2,part,i),dim(2),dim(1))
                label(3,part,i)= -1;
                %fprintf('      A joint is not in the image...\n');
            end
        end
    end

    %Reorder the joints to match pretrained caffe model
    label = reorderJoints(label,order_to_pretrain);
                                                                        %plotImage(image(:,:,:,1),1,14,label(1,:,1),label(2,:,1),1,bbox(1,:),1,label(3,:,1),1,'Original');

    %Resize the image
    [image, label, bbox] = imgScale(image,label,bbox,scale_rate);
                                                                         %plotImage(image(:,:,:,1),1,14,label(1,:,1),label(2,:,1),1,bbox(1,:),1,label(3,:,1),1,'Resize');

    %Rotate the image
    [image, label, bbox] = imgRotate(image, label, bbox,rotate_degree);
                                                                        %plotImage(image(:,:,:,1),1,14,label(1,:,1),label(2,:,1),1,bbox(1,:),1,label(3,:,1),1,'Rotate');

    %crop the image to input box size (default as 368)
    [image, label, bbox] = imgCrop(image, label, bbox, boxsize);
                                                                        %f=1;plotImage(image(:,:,:,f),1,14,label(1,:,f),label(2,:,f),1,bbox(f,:),1,label(3,:,f),1,'Crop');

    %flip the image
    if (do_flip)
        [image, label, bbox] = imgFlip(image, label, bbox, boxsize);
    end
                                                                       %f=1; plotImage(image(:,:,:,f),1,14,label(1,:,f),label(2,:,f),1,bbox(f,:),1,label(3,:,f),1,'Flip');

    %Mark the joints out of bbox by a margin as visibility = -1.
    %You can set the margin inside the function outOfBox().
    [label, max_bbox] = outOfBox(label, bbox);
                                                                        %f=1; plotImage(image(:,:,:,f),1,14,label(1,:,f),label(2,:,f),1,max_bbox(f,:),1,label(3,:,f),1,'outBox');

    %Image preprocessing(Mean, width<->height,RBG->BRG)
    image = preprocess(image);
                                                                        %f=1; plotImage(image(:,:,:,f),0,14,label(1,:,f),label(2,:,f),1,max_bbox(f,:),1,label(3,:,f),1,'preprocess');
    %Preduce label map, and flip width<->height
    labelmap = produceLabelMap(label,boxsize,stride,sigma);
                                                                        %f=1;
                                                                        %plotLabel(labelmap(:,:,:,f),15);
                                                                        %figure('name','Final Image');imshow(image(:,:,:,f),[]);


    centerMap = produceCenterMap([boxsize boxsize], boxsize/2, boxsize/2, sigma_center);
                                                                        %plotCenterMap(centerMap);



end



function plotCenterMap(center_map)
    figure('name','CenterMap');
    imshow(center_map);
end

function plotLabel(labelmap,np)
    figure('name','LabelMap');hold on;
    for i=1:np
        subplot(4,4,i);imshow(labelmap(:,:,i));
    end
end

function plotImage(img,ori,np,x,y,plotJoint, box,plotBbox,visibility,plotAll,figureName)

    figure('name',figureName);
    %plot image
    if ori
        imshow(img);
    else
        imshow(img,[]);
    end
    hold on;

    %plot Joints
    if plotJoint
        for i=1:np
            if plotAll
                if(visibility(i) ==1)
                    plot(x(i),y(i),'rx');
                elseif(visibility(i) ==0)
                    plot(x(i),y(i),'gx');
                else
                    plot(x(i),y(i),'bx');
                end
            else
                if(visibility(i)==1)
                    plot(x(i),y(i),'rx');
                end
            end
        end
    end

    %plot bbox
    if plotBbox
        %%plot vertice
        plot(box(1),box(2),'y*');
        plot(box(3),box(4),'y*');
        plot(box(3),box(2),'y*');
        plot(box(1),box(4),'y*');

        %%plot edge
        boxx1 = [box(1),box(3)];boxy1 = [box(2),box(2)];plot(boxx1,boxy1,'y');
        boxx2 = [box(1),box(1)];boxy2 = [box(2),box(4)];plot(boxx2,boxy2,'y');
        boxx3 = [box(3),box(3)];boxy3 = [box(2),box(4)];plot(boxx3,boxy3,'y');
        boxx4 = [box(1),box(3)];boxy4 = [box(4),box(4)];plot(boxx4,boxy4,'y');
    end
end

function centerMap = produceCenterMap(im_size, x, y, sigma)
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    centerMap = exp(-Exponent);
    centerMap(centerMap < 0.01 ) = 0;
    centerMap(centerMap >1 ) = 1;
end

function label_map = produceLabelMap(label_flip,boxsize,stride,sigma)
    label_size = boxsize / stride;
    np = size(label_flip,2);
    numSeq = size(label_flip,3);
    label_map = zeros([label_size label_size np+1 numSeq]);

    start = stride / 2.0 -0.5;
    for k = 1:numSeq
        for i=1:np
            if(label_flip(3,i,k) >=0 )
                center_x = label_flip(1,i,k);
                center_y = label_flip(2,i,k);
                [X,Y] = meshgrid(1:label_size, 1:label_size);
                X = (X-1)*stride + start - center_x;
                Y = (Y-1)*stride + start - center_y;
                D2 = X.^2 + Y.^2;
                Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
                label = exp(-Exponent);
                label(label < 0.01 ) = 0;
                label(label >1 ) = 1;
            else
                label = zeros([label_size,label_size]);
            end
            label_map(:,:,i,k) = permute(label,[2 1]);
        end

        background = ones([label_size,label_size]);
        for m=1:label_size
            for n=1:label_size
                maxV = 0;
                for t =1:np
                    if(maxV < label_map(m,n,t,k))
                        maxV = label_map(m,n,t,k);
                    end
                end
                background(m,n) = max( 1-maxV,0 );
            end
        end
        label_map(:,:,np+1,k) = background;
    end
end

function label_map = produceLabelMap_large(label_flip,boxsize,stride,sigma)
    label_size = boxsize / stride;
    np = size(label_flip,2);
    numSeq = size(label_flip,3);
    label_map = zeros([label_size label_size np+1 numSeq]);

    for k = 1:numSeq
        for i=1:np
            if(label_flip(3,i,k) >=0 )
                center_x = label_flip(1,i,k);
                center_y = label_flip(2,i,k);
                [X,Y] = meshgrid(1:boxsize, 1:boxsize);
                X = X - center_x;
                Y = Y - center_y;
                D2 = X.^2 + Y.^2;
                Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
                label_l = exp(-Exponent);
                label_l(label_l < 0.01 ) = 0;
                label_l(label_l >1 ) = 1;
                label = imresize(label_l,1/stride);
            else
                label = zeros([label_size,label_size]);
            end
            label_map(:,:,i,k) = permute(label,[2 1]);
        end

        background = ones([label_size,label_size]);
        for m=1:label_size
            for n=1:label_size
                maxV = 0;
                for t =1:np
                    if(maxV < label_map(m,n,t,k))
                        maxV = label_map(m,n,t,k);
                    end
                end
                background(m,n) = max( 1-maxV,0 );
            end
        end
        label_map(:,:,np+1,k) = background;
    end
end


function [image_scale, label_scale, bbox_scale] = imgScale(image, label, bbox, scale_rate)
        image_scale = imresize(image , scale_rate);
        label_scale(1:2,:,:) = label(1:2,:,:) .* scale_rate;
        label_scale(3,:,:) = label(3,:,:);
        bbox_scale = bbox .* scale_rate;

end

function [image_rotate, label_rotate, bbox_rotate] = imgRotate(image_scale, label_scale, bbox_scale, rotate_degree)

    numSeq = size(bbox_scale,1);

    rotateM = [cosd(-rotate_degree) -sind(-rotate_degree); sind(-rotate_degree) cosd(-rotate_degree)];

    tform = affine2d([cosd(rotate_degree) -sind(rotate_degree) 0; sind(rotate_degree) cosd(rotate_degree) 0; 0 0 1]);

    % rotate image by rotate_degree 
    image_rotate = imwarp(image_scale,tform,'cubic','FillValues',128);

    %This transformation only apply when -90 < -degree < 0 < max_degree <90
    %Can be modified by more general formulation
    w_offset = 0;
    h_offset = 0;
    if rotate_degree >0
        h_offset = size(image_scale,2)*sind(rotate_degree);
    elseif rotate_degree <0
        w_offset = size(image_scale,1)*sind(-rotate_degree);
    else
    end

    label_rotate = zeros(size(label_scale));
    for k = 1:numSeq
        label_rotate(1:2,:,k) = rotateM * label_scale(1:2,:,k);
    end

    offset_label = repmat([w_offset;h_offset],[1 size(label_scale,2) size(label_scale,3)]);

    label_rotate(1:2,:,:) = label_rotate(1:2,:,:) + offset_label;
    label_rotate(3,:,:) = label_scale(3,:,:);

    box_tran = zeros([4,4,2]);
    box_tran(:,:,1) = [1 1 0 0;0 0 0 0;0 0 1 1;0 0 0 0];
    box_tran(:,:,2) = [0 0 0 0;1 0 1 0;0 0 0 0;0 1 0 1];

    box_full = zeros([2,4,numSeq]);
    bbox_rotate = zeros(size(bbox_scale));
    for k=1:numSeq
        box_full(1,:,k) = bbox_scale(k,:) * box_tran(:,:,1);
        box_full(2,:,k) = bbox_scale(k,:) * box_tran(:,:,2);
        box_full(:,:,k) = rotateM * box_full(:,:,k);

        bbox_rotate(k,1) = min(box_full(1,:,k));
        bbox_rotate(k,2) = min(box_full(2,:,k));
        bbox_rotate(k,3) = max(box_full(1,:,k));
        bbox_rotate(k,4) = max(box_full(2,:,k));
    end
    bbox_rotate = bbox_rotate + repmat([w_offset h_offset],[numSeq 2]);

end

function [image_crop, label_crop, bbox_crop] = imgCrop(image_rotate, label_rotate, bbox_rotate, boxsize)

%To make sure all the box are within the boxsize. Information of joints will not get lost.
maxBox = 0;
for seq = 1:size(bbox_rotate,1)
  if (max([bbox_rotate(seq,3)-bbox_rotate(seq,1),bbox_rotate(seq,4)-bbox_rotate(seq,2)]) > maxBox)
    maxBox = max([bbox_rotate(seq,3)-bbox_rotate(seq,1),bbox_rotate(seq,4)-bbox_rotate(seq,2)]);
  end
end

%Resize directly will not involve any special transformation of box & labels
if( maxBox > boxsize ) %If preprocess image already too large
   smaller_scale = (boxsize / maxBox) / 1.1  ;
   %fprintf('      The multiplier is set too big...resizing to smaller one, by factor %.4f\n',smaller_scale);
   [image_rotate,label_rotate,bbox_rotate] = imgScale(image_rotate,label_rotate,bbox_rotate,smaller_scale);
end


numSeq = size(bbox_rotate,1);
image_crop = uint8 (ones([boxsize,boxsize,3,numSeq])) .* 128;
label_crop = zeros(size(label_rotate));
bbox_crop = zeros(size(bbox_rotate));

    for i=1:numSeq
        center_x = round( (bbox_rotate(i,1) + bbox_rotate(i,3)) / 2 );
        center_y = round( (bbox_rotate(i,2) + bbox_rotate(i,4)) / 2 );

        left = min(center_x-1, boxsize/2-1);
        up   = min(center_y-1, boxsize/2-1);
        right = min(size(image_rotate,2)-center_x, boxsize/2);
        down = min(size(image_rotate,1)-center_y, boxsize/2);

        try
            image_crop(boxsize/2-up:boxsize/2+down, boxsize/2-left:boxsize/2+right,:,i) = image_rotate(center_y-up:center_y+down, center_x-left:center_x+right,:,i);
        catch
            error('      something wrong happens in cropping....\n');
        end


        offset_left = (boxsize/2) - (center_x);
        offset_up = (boxsize/2) - (center_y);

        label_crop(1:2,:,i) = label_rotate(1:2,:,i) + repmat([offset_left;offset_up],[1 size(label_rotate,2)]);
        for j=1:size(label_rotate,2)
            if(isNotOnPlane(label_crop(1,j,i),label_crop(2,j,i), boxsize, boxsize))
                label_crop(3,j,i) = -1;
                %fprintf('      After cropping a joint is not shown on plane....\n');
            else
                label_crop(3,j,i) = label_rotate(3,j,i);
            end
        end
        bbox_crop(i,:) = bbox_rotate(i,:) + repmat([offset_left offset_up],[1 2]);
    end
end

function [image_flip, label_flip, bbox_flip] = imgFlip(image_crop,label_crop,bbox_crop,boxsize)

    image_flip = flip(image_crop,2);

    w = boxsize;

    bbox_flip = bbox_crop;
    bbox_flip(:,1) = repmat(w-1,[size(bbox_crop,1) 1]) - bbox_crop(:,3) ;
    bbox_flip(:,3) = repmat(w-1,[size(bbox_crop,1) 1]) - bbox_crop(:,1) ;

    label_flip = label_crop;
    label_flip(1,:,:) = repmat(w-1,[1 size(label_crop,2) size(label_crop,3)]) - label_crop(1,:,:);

    label_flip = swiftLeftRight(label_flip);
end

function label_flip = swiftLeftRight(label_flip)
    right = [3 4 5 9 10 11];
    left  = [6 7 8 12 13 14];

    for i=1:length(right)
       temp = label_flip(:,left(i),:);

       label_flip(:,left(i),:) = label_flip(:,right(i),:);

       label_flip(:,right(i),:) = temp;

    end
end

function img_out = preprocess(img)
    img_out = (double(img) -128) / 256 ;

    img_out = permute(img_out, [2 1 3 4]);
    img_out = img_out(:,:,[3 2 1],:);
end

function [label_in,max_bbox] = outOfBox(label_in, bbox_in)

    max_bbox = zeros(size(bbox_in));

    for i=1:size(label_in,3)
        %margin = 0;
        margin = 0.2*norm(label_in(1:2,6,i)-label_in(1:2,9,i)); %6:L sho, 9:R hip

        max_bbox(i,:) = bbox_in(i,:) + [-margin -margin margin margin];

        for j=1:size(label_in,2)
            if(isNotInBox(label_in(1,j,i),label_in(2,j,i),max_bbox(i,:)) && label_in(3,j,i)~=-1)
               label_in(3,j) = -1;
               %fprintf('A joint is not in the bounding box....Mask it...\n');
            end
        end
    end

end

function notOn = isNotOnPlane(x,y,wid,height)
    notOn = (x<0.001 || y<0.001 || x>wid || y>height);
end

function notIn = isNotInBox(x,y,bbox)
    notIn = (x<bbox(1) || y<bbox(2) || x>bbox(3) || y>bbox(4));
end

function label_reorder = reorderJoints(label, order_to_pretrain)
    if(size(label,2) ~= length(order_to_pretrain) )
        error('dimension not match....\n');
    else
        label_reorder = zeros(size(label));
    end

    for i=1:length(order_to_pretrain)
        label_reorder(:,i,:) = label(:,order_to_pretrain(i),:);
    end
end

function [x,y] = findMaximum(map)
    [~,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
end
