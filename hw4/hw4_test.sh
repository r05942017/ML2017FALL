#!bin/bash
wget https://www.dropbox.com/s/wij1q71qb85xyd1/dict.model?dl=1
mv dict.model?dl=1 dict.model
wget https://www.dropbox.com/s/04d7kg02k87q1ar/1127.h5?dl=1
mv 1127.h5?dl=1 1127.h5
wget https://www.dropbox.com/s/zxcw11jyz7r8w50/dict.model.wv.syn0.npy?dl=1
mv dict.model.wv.syn0.npy?dl=1 dict.model.wv.syn0.npy
wget https://www.dropbox.com/s/pzi8102taljode7/dict.model.syn1neg.npy?dl=1
mv dict.model.syn1neg.npy?dl=1 dict.model.syn1neg.npy
python hw4_test.py $1 $2