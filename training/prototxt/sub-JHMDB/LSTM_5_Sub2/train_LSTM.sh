#!/usr/bin/env sh
set -e

/usr/local/MATLAB/R2015a/bin/matlab -nodisplay -nosplash -nodesktop -r video_train_JHMDB 2>&1|tee ./prototxt/sub-JHMDB/LSTM_5_Sub2/train.log
