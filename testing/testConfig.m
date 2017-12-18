function param = testConfig()

param.use_gpu = 1;
% GPU device number
GPUdeviceNumber = 0;

% Path of caffe. You can change to your own caffe just for testing
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
param.model(1).description = 'LSTM Pose Estimation on sub-JHMDB dataset, Sub1';
param.model(1).description_short = 'LSTM_JHMDB_Sub1';
param.model(1).trainedModel = '../model/sub-JHMDB/JHMDB_Sub1.caffemodel';
param.model(1).deployFile_1 = '../model/sub-JHMDB/LSTM_deploy1.prototxt';
param.model(1).deployFile_2 = '../model/sub-JHMDB/LSTM_deploy2.prototxt';
param.model(1).boxsize = 368;
param.model(1).padValue = 128;
param.model(1).np = 14;
param.model(1).stride = 8;
param.model(1).testAdd = '../dataset/JHMDB/Sub1/';

param.model(2).description = 'LSTM Pose Estimation on sub-JHMDB dataset, Sub2';
param.model(2).description_short = 'LSTM_JHMDB_Sub2';
param.model(2).trainedModel = '../model/sub-JHMDB/JHMDB_Sub2.caffemodel';
param.model(2).deployFile_1 = '../model/sub-JHMDB/LSTM_deploy1.prototxt';
param.model(2).deployFile_2 = '../model/sub-JHMDB/LSTM_deploy2.prototxt';
param.model(2).boxsize = 368;
param.model(2).padValue = 128;
param.model(2).np = 14;
param.model(2).stride = 8;
param.model(2).testAdd = '../dataset/JHMDB/Sub2/';

param.model(3).description = 'LSTM Pose Estimation on sub-JHMDB dataset, Sub3';
param.model(3).description_short = 'LSTM_JHMDB_Sub3';
param.model(3).trainedModel = '../model/sub-JHMDB/JHMDB_Sub3.caffemodel';
param.model(3).deployFile_1 = '../model/sub-JHMDB/LSTM_deploy1.prototxt';
param.model(3).deployFile_2 = '../model/sub-JHMDB/LSTM_deploy2.prototxt';
param.model(3).boxsize = 368;
param.model(3).padValue = 128;
param.model(3).np = 14;
param.model(3).stride = 8;
param.model(3).testAdd = '../dataset/JHMDB/Sub3/';

param.model(11).description = 'LSTM Pose Estimation on PENN dataset';
param.model(11).description_short = 'LSTM_PENN';
param.model(11).trainedModel = '../model/PENN/LSTM_PENN.caffemodel';
param.model(11).deployFile_1 = '../model/PENN/LSTM_deploy1.prototxt';
param.model(11).deployFile_2 = '../model/PENN/LSTM_deploy2.prototxt';
param.model(11).boxsize = 368;
param.model(11).padValue = 128;
param.model(11).np = 14;
param.model(11).stride = 8;
param.model(11).testAdd = '../dataset/PENN/Penn_test/';




