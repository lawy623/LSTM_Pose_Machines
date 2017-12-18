%%Configuration of Caffe and model settings
function param = trainConfig()

param.use_gpu = 1;
% GPU device number
GPUdeviceNumber = 0;

% Model Description
%1:LSTM Pose model on sub-JHMDB dataset, Sub set 1
%2:LSTM Pose model on sub-JHMDB dataset, Sub set 2
%3:LSTM Pose model on sub-JHMDB dataset, Sub set 3
%11: LSTM Pose model on PENN dataset

% Path of caffe. You can change to your own caffe
caffepath = '../caffe/matlab';

fprintf('You set your caffe in caffePath.cfg at: %s\n', caffepath);
addpath(caffepath);
caffe.reset_all();

if(param.use_gpu)
    fprintf('Setting to GPU mode, using device ID %d\n', GPUdeviceNumber);
    caffe.set_mode_gpu();
    caffe.set_device(GPUdeviceNumber);
else
    fprintf('Setting to CPU mode.\n');
    caffe.set_mode_cpu();
end

%% Model Settings
param.model(1).preModel = './prototxt/preModel/caffemodel_iter_250000.caffemodel';
param.model(1).trainFile = './prototxt/sub-JHMDB/LSTM_5_Sub1/LSTM_train.prototxt';
param.model(1).solverFile = './prototxt/sub-JHMDB/LSTM_5_Sub1/LSTM_solver.prototxt';
param.model(1).deployFile = './prototxt/sub-JHMDB/LSTM_5_Sub1/LSTM_deploy.prototxt';
param.model(1).description = 'LSTM Pose Estimation on sub-JHMDB video,length=5,Subset1';
param.model(1).description_short = 'LSTM_subJHMDB_L5_Sub1';
param.model(1).solverState = './prototxt/sub-JHMDB/LSTM_5_Sub1/caffemodel/caffemodel_iter_50000.solverstate';
param.model(1).boxsize = 368;
param.model(1).padValue = 128;
param.model(1).np = 14;
param.model(1).seqTrain = 5;
param.model(1).batchSize = 4;
param.model(1).stride = 8;
param.model(1).trainAdd = '../dataset/JHMDB/Sub1';

param.model(2).preModel = './prototxt/preModel/caffemodel_iter_250000.caffemodel';
param.model(2).trainFile = './prototxt/sub-JHMDB/LSTM_5_Sub2/LSTM_train.prototxt';
param.model(2).solverFile = './prototxt/sub-JHMDB/LSTM_5_Sub2/LSTM_solver.prototxt';
param.model(2).deployFile = './prototxt/sub-JHMDB/LSTM_5_Sub2/LSTM_deploy.prototxt';
param.model(2).description = 'LSTM Pose Estimation on sub-JHMDB video,length=5,Subset2';
param.model(2).description_short = 'LSTM_subJHMDB_L5_Sub2';
param.model(2).solverState = './prototxt/sub-JHMDB/LSTM_5_Sub2/caffemodel/caffemodel_iter_50000.solverstate';
param.model(2).boxsize = 368;
param.model(2).padValue = 128;
param.model(2).np = 14;
param.model(2).seqTrain = 5;
param.model(2).batchSize = 4;
param.model(2).stride = 8;
param.model(2).trainAdd = '../dataset/JHMDB/Sub2';

param.model(3).preModel = './prototxt/preModel/caffemodel_iter_250000.caffemodel';
param.model(3).trainFile = './prototxt/sub-JHMDB/LSTM_5_Sub3/LSTM_train.prototxt';
param.model(3).solverFile = './prototxt/sub-JHMDB/LSTM_5_Sub3/LSTM_solver.prototxt';
param.model(3).deployFile = './prototxt/sub-JHMDB/LSTM_5_Sub3/LSTM_deploy.prototxt';
param.model(3).description = 'LSTM Pose Estimation on sub-JHMDB video,length=5,Subset3';
param.model(3).description_short = 'LSTM_subJHMDB_L5_Sub3';
param.model(3).solverState = './prototxt/sub-JHMDB/LSTM_5_Sub3/caffemodel/caffemodel_iter_50000.solverstate';
param.model(3).boxsize = 368;
param.model(3).padValue = 128;
param.model(3).np = 14;
param.model(3).seqTrain = 5;
param.model(3).batchSize = 4;
param.model(3).stride = 8;
param.model(3).trainAdd = '../dataset/JHMDB/Sub3';

param.model(11).preModel = './prototxt/preModel/caffemodel_iter_250000.caffemodel';
param.model(11).trainFile = './prototxt/PENN/LSTM_5/LSTM_train.prototxt';
param.model(11).solverFile = './prototxt/PENN/LSTM_5/LSTM_solver.prototxt';
param.model(11).deployFile = './prototxt/PENN/LSTM_5/LSTM_deploy.prototxt';
param.model(11).description = 'LSTM Pose Estimation on PENN video,length=5';
param.model(11).description_short = 'LSTM_PENN_L5';
param.model(11).solverState = './prototxt/PENN/LSTM_5/caffemodel/caffemodel_iter_50000.solverstate';
param.model(11).boxsize = 368;
param.model(11).padValue = 128;
param.model(11).np = 14;
param.model(11).seqTrain = 5;
param.model(11).batchSize = 4;
param.model(11).stride = 8;
param.model(11).trainAdd = '../dataset/PENN/Penn_train/';
