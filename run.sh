#!/bin/bash
echo "mode : $1"
if [[ $1 = "train" ]];then
    CUDA_VISIBLE_DEVICE=3 python config.py --mode train | tee log/train/date +"%%m%d/%R"
elif [[ $1 = "debug" ]];then
    CUDA_VISIBLE_DEVICE=3 python config.py --mode debug | tee log/debug/date +"%%m%d/%R"
fi
