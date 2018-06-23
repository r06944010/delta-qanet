#!/bin/bash
echo "mode : $1"
if [[ $1 = "train" ]];then
    CUDA_VISIBLE_DEVICE=3 python config.py --mode train
elif [[ $1 = "debug" ]];then
    CUDA_VISIBLE_DEVICE=3 python config.py --mode debug
fi
