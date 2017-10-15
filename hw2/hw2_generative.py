from pandas import Series,DataFrame
import pandas as pd
import numpy as np
from numpy.linalg import inv
import sys
X_train=sys.argv[1]
Y_train=sys.argv[2]
inputfilename=sys.argv[3]
outputfilename=sys.argv[4]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
DATA = pd.read_csv(X_train,encoding='big5')
label= pd.read_csv(Y_train,encoding='big5')
feature_num=106
data=DATA


data=pd.concat([data, label], axis=1)
data=data.sort_values("label")  							#排序有無50k

num_with50=np.sum(data['label'])

data_with50=data.tail(num_with50)									#50K資料 7841筆
data_no50=data.head(len(data)-num_with50)							#50K以下資料
data_with50= pd.concat([data_with50], ignore_index=True)	#重排index
data_no50= pd.concat([data_no50], ignore_index=True)		#重排index
data_with50=data_with50.values							#轉成numpy
data_no50=data_no50.values									#轉成numpy
data=data.values



mean_with50=np.zeros(len(data_with50[0,:feature_num]))
mean_no50=np.zeros(len(data_no50[0,:feature_num]))									#製造mean
for i in range(len(data_with50[0,:feature_num])):
	mean_with50[i]=np.sum(data_with50[:,i]) / len(data_with50)
for i in range(len(data_no50[0,:feature_num])):
	mean_no50[i]=np.sum(data_no50[:,i])	/ len(data_no50)


covariance_with50=np.zeros((feature_num,feature_num))
covariance_no50=np.zeros((feature_num,feature_num))											#製造covariance
for i in range(len(data_with50)):	
	covariance_with50=covariance_with50+np.array(np.asmatrix(data_with50[i,:feature_num]-mean_with50).T*np.asmatrix(data_with50[i,:feature_num]-mean_with50))
for i in range(len(data_no50)):	
	covariance_no50=covariance_no50+np.array(np.asmatrix(data_no50[i,:feature_num]-mean_no50).T*np.asmatrix(data_no50[i,:feature_num]-mean_no50))
covariance_with50=covariance_with50/len(data_with50)
covariance_no50=covariance_no50/len(data_no50)
covariance=len(data_with50)/len(data)*covariance_with50 + len(data_no50)/len(data)*covariance_no50

w=np.asmatrix(mean_with50-mean_no50)*np.asmatrix(inv(covariance))		#1D matrix
b=-0.5*np.asmatrix(mean_with50)*np.asmatrix(inv(covariance))*np.asmatrix(mean_with50).T+0.5*np.asmatrix(mean_no50)*np.asmatrix(inv(covariance))*np.asmatrix(mean_no50).T+np.log(len(data_with50)/len(data_no50))

y_pred=np.zeros(len(data))

for i in range(len(data)):
	z=np.dot(np.array(w),data[i,0:feature_num])+b
	y_pred[i]=sigmoid(z)


y_pred[y_pred>=0.5]=1
y_pred[y_pred<0.5]=0
count=0
for i in range(len(data)):
	if y_pred[i]==data[i,feature_num]:
		count=count+1
acc=count/len(data)


#------------------------------------------------------------------------------------------------------output
testDATA=pd.read_csv(inputfilename,encoding='big5')

testDATA=testDATA.values
y_pred_test=np.zeros(len(testDATA))
for i in range(len(testDATA)):
	z=np.dot(np.array(w),testDATA[i,0:feature_num])+b
	y_pred_test[i]=sigmoid(z)
y_pred_test[y_pred_test>=0.5]=1
y_pred_test[y_pred_test<0.5]=0


a=(list(range(len(y_pred_test))))
a=np.array(a)+1
out={'id':a,'label':y_pred_test}

output=DataFrame(data=out)
output=output.astype(int)
output.to_csv(outputfilename,index=None)








'''
for i in range(106):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(data[:,i],lable)
	plt.show()
'''