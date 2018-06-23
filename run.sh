#!/bin/bash

NOW=$(date +"%m%d-%R")
echo "mode : $1"
if [[ $1 = "train" ]];then
    CUDA_VISIBLE_DEVICES=1 python config.py --mode train | tee log/$NOW
elif [[ $1 = "debug" ]];then
    CUDA_VISIBLE_DEVICES=1 python config.py --mode debug | tee log/$NOW
fi
