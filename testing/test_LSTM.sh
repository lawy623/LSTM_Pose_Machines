#!/usr/bin/env sh
set -e

/usr/local/MATLAB/R2015a/bin/matlab -nodisplay -nosplash -nodesktop -r benchmark -logfile result/LSTM_PENN.txt #result/LSTM_JHMDB_Sub1.txt 

