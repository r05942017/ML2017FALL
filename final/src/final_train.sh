#!/bin/bash
wget -O chinese_embedding.model.syn0.npy https://www.dropbox.com/s/3x5zirhboncodk8/chinese_embedding.model.syn0.npy?dl=1
python3 final_train.py $1 $2