#!/bin/bash
NUM_PROC=$1
PORT=$2
shift
torchrun --nproc_per_node=$NUM_PROC --master_port $PORT train.py "$@"