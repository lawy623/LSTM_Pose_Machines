#!/usr/bin/env sh
set -e

/usr/local/MATLAB/R2015a/bin/matlab -nodisplay -nosplash -nodesktop -r video_train_PENN 2>&1|tee ./prototxt/PENN/LSTM_5/train.log
