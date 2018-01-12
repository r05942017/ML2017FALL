import os
from skimage import data, io, filters
import numpy as np
import sys


file=sys.argv[1]
read_image=sys.argv[2]
total_file=[]
for dirPath, dirNames, fileNames in os.walk(file):
	total_file=fileNames
total_picture=np.zeros((600*600*3,len(total_file)))

for i in range(len(total_file)):
	filename_buf=file+'/'+total_file[i]
	a=io.imread(filename_buf)
	total_picture[:,i]=a.reshape(600*600*3)

mean=np.mean(total_picture,axis=1)

minus=np.zeros((600*600*3,len(total_file)))							
for i in range(len(total_picture[0])):
	minus[:,i]=total_picture[:,i]-mean			#簡平均

u,s,v=np.linalg.svd(minus,full_matrices=False)	#SVD
eigenvalue=np.power(s,2)

del total_picture,minus

a=io.imread(read_image)
total_picture=a.reshape(600*600*3)

minus=total_picture-mean

	
weight=np.dot(minus,u)
#print('weight=',weight[:4])
test=mean+np.dot(weight[:4],u[:,:4].T)
test-=np.min(test)
test/=np.max(test)
test=test*255
test=test.astype(np.uint8).reshape((600,600,3))	

print(test.shape)

io.imsave('reconstruction.jpg',test)

