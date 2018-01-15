%%This is the code for visualizing the results on PENN test set, make sure
%%you have already run the test script for PENN test.
clear all;
prediction_file = load('../predicts/LSTM_PENN.mat');
prediction = prediction_file.prediction;
clear prediction_file;

% Turn this to 1 if you want only the visible points
visibleOnly = 0;

% We make the visualization on the first 5 videos 
test = [1 2 3 4 5];

% You can use this to randomly produce some results. Do not run too many
% clips at a single time. Otherwise it is easy to run out of memory.
%full_set =1:1068;
%test = datasample(full_set,10,'Replace',false);
%test = sort(test);

testset =test;
length = length(testset);
oldId = 1;
fprintf('loading set 1\n');
data = load(['../../dataset/PENN/Penn_test/testSet' num2str(oldId) '.mat']);

for j = 1:length
    seq = testset(j);
    fprintf('Sequence %d (%d/%d)...\n',seq,j,length);
    nframes = prediction.sequences{seq}.nframes;
    imagedir = prediction.sequences{seq}.frameAdd;
    labeldir = prediction.sequences{seq}.labelAdd;
    predLabel = prediction.sequences{seq}.predLabel;
    label = prediction.sequences{seq}.label;
    bbox = prediction.sequences{seq}.bbox;
    
    if exist(strcat('./videos/Seq',num2str(seq,'%04d'),'_Pred_LSTM.avi'),'file') ~= 0
        fprintf('already there...skip...\n');
        continue;
    end    
    
    videoName = strcat('./videos/Seq',num2str(seq,'%04d'),'_Pred_LSTM.avi');
    v = VideoWriter(videoName);
    v.FrameRate = 12; % modify this if you want different speed 
    open(v);
    
    setId = floor((seq-1)/89+1);
    fprintf('For set %d...\n',setId);
    if(setId ~= oldId)
        oldId = setId;
        fprintf('loading set %d\n',setId);
        clear data;
        data = load(['../../dataset/PENN/Penn_test/testSet' num2str(setId) '.mat']);
    end     

    for i = 1:nframes
        idx = seq - 89*(setId-1);
        imageToMovie = data.test.sequence{idx}.frame{i}.image;

        predLabel_ = predLabel(:,:,i);
        label_ = label(:,:,i);
        bbox_ = bbox(i,:);

        %avoid wrong boxes : some labels are wrong
        if( (bbox_(3)-bbox_(1))==0 && (bbox_(4)-bbox_(2))==0  )
            bbox(i,:) = bbox(i-1,:);
            bbox_ = bbox(i,:);
        end    

        f=figure('Name',num2str(i),'Visible','off');imshow(imageToMovie);hold on;

        limbs = [2 4; 4 6; 3 5; 5 7; 2 3; 9 11; 11 13; 8 10; 10 12;8 9];
        
        if(visibleOnly)
           visJoint = label_(3,:); 
           predLabel_(:,visJoint==0) = nan;
        end    
            
        %%plot position of joints
        for k=1:size(predLabel_,2)
            if(~isnan(predLabel_(1,k)) && ~isnan(predLabel_(2,k)))
                plot(predLabel_(1,k),predLabel_(2,k),'gx');
            end    
        end

        %%plot vertice 
        plot(bbox_(1),bbox_(2),'y*');plot(bbox_(3),bbox_(4),'y*');
        plot(bbox_(3),bbox_(2),'y*');plot(bbox_(1),bbox_(4),'y*');

        %%plot edge
        boxx1 = [bbox_(1),bbox_(3)];boxy1 = [bbox_(2),bbox_(2)];plot(boxx1,boxy1,'y');
        boxx2 = [bbox_(1),bbox_(1)];boxy2 = [bbox_(2),bbox_(4)];plot(boxx2,boxy2,'y');
        boxx3 = [bbox_(3),bbox_(3)];boxy3 = [bbox_(2),bbox_(4)];plot(boxx3,boxy3,'y');
        boxx4 = [bbox_(1),bbox_(3)];boxy4 = [bbox_(4),bbox_(4)];plot(boxx4,boxy4,'y');

        %put limbs
        facealpha = 0.6;
        labelLimb = predLabel_;
        colors = hsv(size(limbs,1));

        for p = 1:size(limbs,1)
            X = labelLimb(1,limbs(p,:));
            Y = labelLimb(2,limbs(p,:));

            if(~sum(isnan(X)))
                a = 1/2 * sqrt((X(2)-X(1))^2+(Y(2)-Y(1))^2);
                maxL = max(bbox_(4)-bbox_(2),bbox_(3)-bbox_(1) );
                b = (maxL)/40;
                t = linspace(0,2*pi);
                XX = a*cos(t);
                YY = b*sin(t);
                w = atan2(Y(2)-Y(1), X(2)-X(1));
                x = (X(1)+X(2))/2 + XX*cos(w) - YY*sin(w);
                y = (Y(1)+Y(2))/2 + XX*sin(w) + YY*cos(w);
                h = patch(x,y,colors(p,:));
                set(h,'FaceAlpha',facealpha);
                set(h,'EdgeAlpha',0);
            end
        end
        try
            labelLimb = labelLimb(:,~isnan(labelLimb(1,:)));
            plot(labelLimb(1,:), labelLimb(2,:), 'k.', 'MarkerSize', (maxL)/20);
        catch
            error('not valid');
        end

        %write video
        M_pre = getframe;
        writeVideo(v,M_pre);
        fprintf(' Frame %d/%d ...\n',i,nframes);
    end
    clear M_pre;
    close(v);clear v;
end



    








