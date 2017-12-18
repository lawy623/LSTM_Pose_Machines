close all;
addpath('src'); 

%set the GPU device
g = gpuDevice(1);

param = testConfig();

%% choose a model
% 1:LSTM Pose Model. Trained on JHMDB Dataset ( Sub1 )
% 2:LSTM Pose Model. Trained on JHMDB Dataset ( Sub2 )
% 3:LSTM Pose Model. Trained on JHMDB Dataset ( Sub3 )
% 11:LSTM Pose Model. Trained on PENN Dataset         

benchmark_modelID = 11;   

%% run benchmark
if(benchmark_modelID == 11)
    prediction_file = run_benchmark_GPU_PENN(param, benchmark_modelID);
elseif(benchmark_modelID == 1 || benchmark_modelID == 2 || benchmark_modelID == 3)
    prediction_file = run_benchmark_GPU_JHMDB(param, benchmark_modelID);
else
    error('Please check the model ID selection...');
end

fprintf('prediction file saved at %s\n', prediction_file);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
