#!/bin/bash

NOW=$(date +"%m%d-%R")
echo "CUDA_VISIBLE_DEVICES=$3 python config.py $2 --mode $1 | tee log/$2/$NOW"
echo "  GPU : $3"
echo "  $1"
echo "  embedding type : $2"

# CUDA_VISIBLE_DEVICES=$3 python config.py $2 --mode $1
CUDA_VISIBLE_DEVICES=$3 python config.py $2 --mode $1 | tee -a log/$2/$1.$NOW

#CUDA_VISIBLE_DEVICES=0 python config.py all --mode train | tee -a log/all/$(date +"%m%d-%R")
#CUDA_VISIBLE_DEVICES=1 python config.py char --mode train | tee -a log/char/$(date +"%m%d-%R")
#CUDA_VISIBLE_DEVICES=2 python config.py char_800 --mode train | tee -a log/char_800/$(date +"%m%d-%R")
