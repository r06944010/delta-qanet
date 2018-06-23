#!/bin/bash

NOW=$(date +"%m%d-%R")
echo "CUDA_VISIBLE_DEVICES=$3 python config.py $2 --mode $1 | tee log/$2/$NOW"
echo "  GPU : $3"
echo "  $1"
echo "  embedding type : $2"
CUDA_VISIBLE_DEVICES=$3 python config.py $2 --mode $1 --type $2 | tee log/$2/$NOW
