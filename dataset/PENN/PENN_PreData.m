%%This is the code for preparing PENN Train && Test sets.%%

%Create a Summary file, which contains all the labels.
labelFolder= ('./Penn_Action/labels/');
frameFolder= ('./Penn_Action/frames/');
numSeq = 2326;

summary.train = zeros(numSeq,1);
summary.nframes = zeros(numSeq,1);
summary.dim = zeros(numSeq,2);
fprintf('Writing Summary...\n');

for i = 1:numSeq
    fprintf('Processing squence num %d/%d ...\n', i, numSeq);
    squenceInd = strcat ( num2str(i,'%04d'), '.mat');
    labelAdd = strcat ( labelFolder, squenceInd);
    data = load( labelAdd );
    
    summary.frameAdd{i,1} = strcat ( frameFolder, num2str(i,'%04d'));
    summary.labelAdd{i,1} = labelAdd;
    summary.train(i) = data.train;
    summary.nframes(i) = data.nframes;
    summary.action{i,1} = data.action;
    summary.pose{i,1} = data.pose;
    summary.dim(i,1) = data.dimensions(1);
    summary.dim(i,2) = data.dimensions(2);
    summary.bbox{i,1} = data.bbox;
    summary.x{i,1} = data.x;
    summary.y{i,1} = data.y;
    summary.visibility{i,1} = data.visibility;
end    

%Mistake in the dataset. Bboxes of the last frame of sequence of 1865 and 1154 are missing.
%Manually add the bbox using the previous frame.
summary.bbox{1865}(73,:)=summary.bbox{1865}(72,:);
summary.bbox{1154}(72,:)=summary.bbox{1154}(71,:);

summary.trainInd = find(summary.train == 1);
summary.testInd = find(summary.train == -1);
numTrain = length(summary.trainInd);
summary.randomTrainInd = datasample(summary.trainInd,numTrain,'Replace',false); %randomly order the train set.

fileName = sprintf('./summary.mat');
save(fileName, 'summary','-v7.3');

%Prepare Training Dataset
randomTrainInd = summary.randomTrainInd;
numTrain = length(randomTrainInd);
numSeqInSet =74;

mkdir('./Penn_train');
fprintf('Writing Training Sets...\n');

for i = 1:(numTrain/numSeqInSet) %should be 17 sets in total
    for j =1:numSeqInSet
        ind = (i-1)*numSeqInSet +j;
        seqId = randomTrainInd(ind); %Real ID, help to identify the sequence.
        train.sequence{j,1}.seqId = seqId;
        train.sequence{j,1}.train = summary.train(seqId);
        train.sequence{j,1}.action = summary.action{seqId};
        train.sequence{j,1}.pose = summary.pose{seqId};
        train.sequence{j,1}.frameAdd = summary.frameAdd{seqId};
        train.sequence{j,1}.labelAdd = summary.labelAdd{seqId};
        train.sequence{j,1}.dim = summary.dim(seqId,:) ;
        train.sequence{j,1}.nframes = summary.nframes(seqId);
        
        frameFolder = train.sequence{j}.frameAdd;
        for k = 1:train.sequence{j}.nframes
            imgInd = strcat( num2str(k,'%06d'), '.jpg');
            imgAdd = strcat(frameFolder ,'/', imgInd);
            try
                img = imread(imgAdd);
            catch
                error('Image cannot be loaded, make sure you have %s', imgAdd);
            end
            train.sequence{j}.frame{k,1}.address = imgAdd;
            train.sequence{j}.frame{k,1}.image = img;
            train.sequence{j}.frame{k,1}.bbox = summary.bbox{seqId}(k,:);
            train.sequence{j}.frame{k,1}.label.x = summary.x{seqId}(k,:);
            train.sequence{j}.frame{k,1}.label.y = summary.y{seqId}(k,:);
            train.sequence{j}.frame{k,1}.label.visibility = summary.visibility{seqId}(k,:);
        end    
    end
    train.numSeqInSet = numSeqInSet;
    train.totalTrain = numTrain;
    train.numOfSet = numTrain/numSeqInSet;
    
    fileName = sprintf('Penn_train/trainSet%d.mat',i);
    save(fileName, 'train','-v7.3');
    fprintf('Creation over for Set %d/%d...\n',i,numTrain/numSeqInSet);
    clear train;
end    

%Prepare Training Dataset
testInd = summary.testInd;
numTest = length(testInd);
numSeqInSet = 89;
mkdir('./Penn_test');
fprintf('Writing Test Sets...\n');

for i = 1:(numTest/numSeqInSet) %should be 12 testing sets in total
    for j =1:numSeqInSet
        ind = (i-1)*numSeqInSet +j;
        seqId = testInd(ind);  %Real ID, help to identify the sequence.
        test.sequence{j,1}.seqId = seqId;
        test.sequence{j,1}.train = summary.train(seqId);
        test.sequence{j,1}.action = summary.action{seqId};
        test.sequence{j,1}.pose = summary.pose{seqId};
        test.sequence{j,1}.frameAdd = summary.frameAdd{seqId};
        test.sequence{j,1}.labelAdd = summary.labelAdd{seqId};
        test.sequence{j,1}.dim = summary.dim(seqId,:) ;
        test.sequence{j,1}.nframes = summary.nframes(seqId);
        
        frameFolder = test.sequence{j}.frameAdd;
        for k = 1:test.sequence{j}.nframes
            imgInd = strcat( num2str(k,'%06d'), '.jpg');
            imgAdd = strcat(frameFolder ,'/', imgInd);
            try
                img = imread(imgAdd);
            catch
                error('image cannot be loaded, make sure you have %s', imgAdd);
            end
            test.sequence{j}.frame{k,1}.address = imgAdd;
            test.sequence{j}.frame{k,1}.image = img;
            test.sequence{j}.frame{k,1}.bbox = summary.bbox{seqId}(k,:);
            test.sequence{j}.frame{k,1}.label.x = summary.x{seqId}(k,:);
            test.sequence{j}.frame{k,1}.label.y = summary.y{seqId}(k,:);
            test.sequence{j}.frame{k,1}.label.visibility = summary.visibility{seqId}(k,:);
        end    
    end
    test.numSeqInSet = numSeqInSet;
    test.totalTrain = numTest;
    test.numOfSet = numTest/numSeqInSet;
    
    fileName = sprintf('Penn_test/testSet%d.mat',i);
    save(fileName, 'test','-v7.3');
    fprintf('Creation over for Set %d/%d...\n',i,numTest/numSeqInSet);
    clear test;
end    

