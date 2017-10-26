from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import sys

X_train=sys.argv[1]
Y_train=sys.argv[2]
inputfilename=sys.argv[3]
outputfilename=sys.argv[4]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
DATA = pd.read_csv(X_train)
label= pd.read_csv(Y_train)
feature_num=106
data=DATA

data=data.values

label=label.values
batch_size=4000
data_size=len(data)
b=0
w=np.zeros(feature_num)
y_pred=np.zeros(len(data))

lr=10	#learning rate
iteration=10
lr_b=1
lr_w=np.zeros(feature_num)+1


f=0
j=0
run=1
z=0

test=np.zeros(feature_num)+1
a=np.asmatrix(data)*np.asmatrix(test).T
a=np.array(a)[:,0]



w_best=np.zeros(feature_num)
b_best=0
acc_best=0
for i in range(iteration):

	b_grade=0.0
	w_grade=np.zeros(feature_num)


	while run==1:


		z=np.dot(w,data[j,:])+b
		if z<-100:
			z=-100
		f=sigmoid(z)
		equ=(label[j]-f)

		b_grade=b_grade-equ*1.0
		w_grade[:]=w_grade[:]-equ*data[j,:]


		if j>=data_size-1:
			j=0
		if (j+1)%batch_size==0:
			j=j+1
			break		
		j=j+1



	lr_b=lr_b+b_grade**2
	b=b-lr/np.sqrt(lr_b)*b_grade
	for j in range(feature_num):
		lr_w[j]=lr_w[j]+w_grade[j]**2
		w[j]=w[j]-lr/np.sqrt(lr_w[j])*w_grade[j]

	for j in range(len(data)):
		z=np.dot(np.array(w),data[j,:])+b
		if z<-100:
			z=-100
		y_pred[j]=sigmoid(z)
	if np.isnan(f):

		break


	y_pred[y_pred>=0.5]=1
	y_pred[y_pred<0.5]=0
	count=0
	for j in range(len(data)):
		if y_pred[j]==label[j]:
			count=count+1
	acc=count/len(data)

	if acc>acc_best:
		acc_best=acc
		w_best=w
		b_best=b



#------------------------------------------------------------------------------------------------------output
testDATA=pd.read_csv(inputfilename)

testDATA=testDATA.values
y_pred_test=np.zeros(len(testDATA))
for i in range(len(testDATA)):
	z=np.dot(np.array(w_best),testDATA[i,0:feature_num])+b_best
	if z<-100:
		z=-100
	y_pred_test[i]=sigmoid(z)
y_pred_test[y_pred_test>=0.5]=1
y_pred_test[y_pred_test<0.5]=0


a=(list(range(len(y_pred_test))))
a=np.array(a)+1
out={'id':a,'label':y_pred_test}

output=DataFrame(data=out)
output=output.astype(int)
output.to_csv(outputfilename,index=None)






