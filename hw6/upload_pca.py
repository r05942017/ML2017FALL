import os
from skimage import data, io, filters
import numpy as np
import sys



imagename=sys.argv[2]
a=io.imread('num1.jpg')
total_picture=a.reshape(600*600*3)

#mean=np.mean(total_picture,axis=1)
mean_load=np.load('mean.npz')
mean=mean_load['mean']
minus=total_picture-mean

npzfile = np.load('SVD.npz')
u=npzfile['u']

	
weight=np.dot(minus,u)
#print('weight=',weight[:4])
test=mean+np.dot(weight[:4],u[:,:4].T)
test-=np.mean(test)
test/=np.max(test)
test=test*255
test=test.astype(np.uint8).reshape((600,600,3))	

print(test.shape)

io.imsave('reconstruction.jpg',test)
