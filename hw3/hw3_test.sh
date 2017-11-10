#!/bin/bash 
wget https://www.dropbox.com/s/fxiqd487zel0voh/hw3_model.h5?dl=1
mv hw3_model.h5?dl=1%0D hw3_model.h5
python3 hw3_test.py $1 $2
