from pandas import Series,DataFrame
import pandas as pd
import numpy as np
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

data = pd.read_csv('train.csv',encoding='big5', usecols=lambda x: x not in ['日期','測站','測項'])
data=DataFrame(data)

datalenth=len(data)/18
AM_TEMP=data[0:18]

AM_TEMP=AM_TEMP.rename(index={0:'AMB_TEMP',1:'CH4',2:'CO',3:'NMHC',4:'NO',5:'NO2',6:'NOx',7:'O3',8:'PM10',9:'PM2.5',
						10:'RAINFALL',11:'RH',12:'SO2',13:'THC',14:'WD_HR',15:'WIND_DIREC',16:'WIND_SPEED',17:'WS_HR'})

for i in range(1,int(len(data)/18),1):      #把相同item排在同一列
	j=18*i
	buf=data[j:j+18]
	buf=buf.rename(index={j:'AMB_TEMP',j+1:'CH4',j+2:'CO',j+3:'NMHC',j+4:'NO',j+5:'NO2',j+6:'NOx',j+7:'O3',j+8:'PM10',j+9:'PM2.5',
						j+10:'RAINFALL',j+11:'RH',j+12:'SO2',j+13:'THC',j+14:'WD_HR',j+15:'WIND_DIREC',j+16:'WIND_SPEED',j+17:'WS_HR'})
	AM_TEMP = pd.concat([AM_TEMP,buf],axis=1,ignore_index=True)
AM_TEMP=AM_TEMP.T   #矩陣旋轉

for i in range(0,int(len(AM_TEMP)),1):   #change NR to 0
	if AM_TEMP['RAINFALL'][i]=='NR':
		AM_TEMP['RAINFALL'][i]=0

#----------------------------------------------------------------------------------------------
num=9
b=0
w=np.zeros((num,2))
lr=5	#learning rate
iteration=50000
lr_b=0
lr_w=np.zeros((num,2))
lambd=0

AM_TEMP=AM_TEMP.apply(pd.to_numeric)    #str to float
y_correct=DataFrame()
for cor in range(12):				#get y_correct
	if cor==0:
		y_correct=AM_TEMP['PM2.5'][num:480-120]
	else:
		y_correct=y_correct.append( AM_TEMP['PM2.5'][cor*480+num:(cor+1)*480-120], ignore_index=True)

run=1
val=AM_TEMP.values
val2=val
AM_TEMP=AM_TEMP.values


for i in range(iteration):

	b_grade=0.0
	w_grade=np.zeros((num,3))

	j=0
	print(i)
	while run==1:
		equ=2.0*(AM_TEMP[j+num,9]-b-np.dot(w[:,0],AM_TEMP[j:j+num,9])-np.dot(w[:,1],np.square(AM_TEMP[j:j+num,9])))	#Stochastic Gradient Descent
		b_grade=b_grade-equ*1.0

		for k in range(num):
			w_grade[k,0]=w_grade[k,0]-equ*AM_TEMP[j+k,9]
			w_grade[k,1]=w_grade[k,1]-equ*np.square(AM_TEMP[j+k,9])

		if (j+num+1)%360==0:
			j=j+num+120		
		j=j+1
		if j>=(5760-num):
			break

	for j in range(num):						#regulation
		w_grade[j][0]=w_grade[j][0]+2*lambd*w[j][0]
		w_grade[j][1]=w_grade[j][1]+2*lambd*w[j][1]

	lr_b=lr_b+b_grade**2
	b=b-lr/np.sqrt(lr_b)*b_grade

	for a in range(num):
		lr_w[a,0]=lr_w[a,0]+w_grade[a,0]**2
		w[a,0]=w[a,0]-lr/np.sqrt(lr_w[a,0])*w_grade[a,0]
		lr_w[a,1]=lr_w[a,1]+w_grade[a,1]**2
		w[a,1]=w[a,1]-lr/np.sqrt(lr_w[a,1])*w_grade[a,1]

	y_pred=np.zeros(5760-12*num)
	for cor in range(12):
		buf=b

		for x in range(num):
			buf=buf+AM_TEMP[cor*480+x:((cor+1)*480-num+x)-120,9]*w[x,0]+np.square(AM_TEMP[cor*480+x:((cor+1)*480-num+x)-120,9])*w[x,1]
		if cor==0:
			y_pred=buf
		else:
			y_pred=np.hstack((y_pred,buf))
	y_pred=y_pred+lambd*np.sum(np.square(w))				#regulation

	pred_rmse=rmse(y_pred,y_correct.values)
	print('rmse=',float(pred_rmse))


#-------------------------------------------------------------------------------------------------val2

y_validation=DataFrame()
for cor in range(12):				#get y_validation2
	if cor==0:
		y_validation=val[436+num:480,9]
	else:
		y_validation=np.hstack((y_validation,val[436+cor*480+num:480+cor*480,9]))


for cor in range(12):
	buf=b
	for x in range(num):
		buf=buf+val[cor*480+x+436:cor*480+x+480-num,9]*w[x,0]+np.square(AM_TEMP[cor*480+x+436:cor*480+x+480-num,9])*w[x,1]
	if cor==0:
		y_test=buf
	else:
		y_test=np.hstack((y_test,buf))
y_test=y_test+lambd*np.sum(np.square(w))

vali_rmse2=rmse(y_test,y_validation)
print('val_rmse2=',vali_rmse2)


