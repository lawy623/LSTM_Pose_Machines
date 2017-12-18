#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt -gpu 1 $@ 

##Run the complied caffe program, train mode, set solver function(In side it defines the network)##
