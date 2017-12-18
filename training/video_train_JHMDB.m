%%Training script for training on sub-JHMDB dataset

%add paths
addpath('./src');
      
param = trainConfig();

%ID = 1/2/3: LSTM Pose Model on Sub-JHMDB dataset. Each ID represent a
%different subset. Please specify it before training.
modelID = 1;
model = param.model(modelID);

%train data address
trainAdd = model.trainAdd;

%Choose to finetune or resume
contiTrain = 0;
if(contiTrain == 0)
    solver = caffe.Solver(model.solverFile);
    preModel = model.preModel;
    solver.net.copy_from(preModel);
else    
    %resume training from previous
    solver = caffe.Solver(model.solverFile);
    solver.restore(model.solverState);
end

%model & transformation params
batch_size = model.batchSize;
seqTrain = model.seqTrain;
nPart = model.np;
boxsize= model.boxsize;
stride =model.stride;

%reading Training Data
data = '/train.mat';
trainDataAdd = strcat(trainAdd,data);
fprintf('Loading Training Dataset: %s...\n',trainDataAdd);
trainData  = load(trainDataAdd);
train = trainData.train;
clear trainData;

seqLength = length(train.sequences);
maxIter = 2000;  %max num of epoch.

startIter= floor(solver.iter()*batch_size/seqLength)+1;

for iteration = startIter:maxIter
%random permutation to get the data, increase randomness
order = randperm(seqLength);

    for batch = 1:(floor(seqLength/batch_size))
        %fprintf('  Processing iteration %d for Training....\n',seqLength/batch_size*(i-1)+batch ); 
        input = cell(batch_size,1);
        label = cell(batch_size,1);
        centerMap = cell(batch_size,1);
        trainBatch = cell(batch_size,1);
        trainInd = zeros(1,batch_size);

        for n = 1:batch_size
            trainInd(n) = order( (batch-1)*batch_size + n);
            trainBatch{n} = train.sequences{trainInd(n)};
            %check whether using training or not
            if trainBatch{n}.train~=1
                error('Error in train Sequence. Not Belong to Train Set.\n');
            end
           
            %seperately conduct transformation. Keep transformation in a
            %sequence consistent, but random in diff sequence.    
            [input{n},label{n},centerMap{n}] = transformation_JHMDB(trainBatch{n},boxsize,stride,nPart,seqTrain);      
        end

        %Restruct the data from different sequences in a batch
        %And send them to the net's inputs
        data_ = cell(seqTrain,1);
        label_ = cell(seqTrain,1);
        centerMap_ = zeros([boxsize boxsize 1 batch_size],'single');
        for j = 1:seqTrain
            for k = 1:batch_size
                data_{j}(:,:,:,k) = input{k}(:,:,:,j);
                label_{j}(:,:,:,k) = label{k}(:,:,:,j);
            end
            dataStr = strcat( 'data', num2str(j));
            labelStr = strcat( 'label',num2str(j));
            solver.net.blobs(dataStr).set_data(single(data_{j}));
            solver.net.blobs(labelStr).set_data(single(label_{j}));
        end
        for t = 1:batch_size
            centerMap_(:,:,:,t) = centerMap{t}(:,:,:);
        end
        solver.net.blobs('center_map').set_data(single(centerMap_))

        solver.step(1);
        iter = solver.iter();
    end
end  
