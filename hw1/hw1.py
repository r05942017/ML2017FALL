from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import	sys

inputfilename=sys.argv[1]
outputfilename=sys.argv[2]

data = np.genfromtxt(inputfilename, delimiter=',')
num=9

b= 0.284144114677 
w= np.array([[  6.26997386e-02 , -2.02904957e-03],
 [  3.43654482e-02 ,  2.39488862e-04],
 [  3.77818114e-02 ,  2.98879764e-03],
 [ -1.92986678e-02 , -4.44706460e-03],
 [ -2.79300315e-02 ,  1.11788425e-03],
 [  7.51884722e-02 ,  4.29073332e-03],
 [ -7.81288775e-02 , -5.10310266e-03],
 [  2.48224671e-01 , -1.61053713e-03],
 [  6.73992327e-01 ,  4.02240497e-03]])

lambd=0
data=DataFrame(data)
data=data.drop(data.columns[0:2],axis=1)

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

print('am=',AM_TEMP)
AM_TEMP=AM_TEMP.T   #矩陣旋轉
a=AM_TEMP['RAINFALL'][0]
print('a=',a)
for i in range(0,int(len(AM_TEMP)),1):   #change NR to 0
	if np.isnan(AM_TEMP['RAINFALL'][i]):
		print('work')
		AM_TEMP['RAINFALL'][i]=0
#--------------------------------------------------------------------------------
y_pred=np.zeros(240)

for cor in range(240):

	buf=b+np.dot(AM_TEMP['PM2.5'][cor*9+9-num:((cor+1)*9)],w[:,0])+np.dot(np.square(AM_TEMP['PM2.5'][cor*9+9-num:((cor+1)*9)]),w[:,1])
	print(buf)

	y_pred[cor]=buf

out={'value':y_pred}
output=DataFrame(data=out)
output=output.apply(pd.to_numeric)  
output=output.rename_axis('id')
for i in range(int(len(AM_TEMP)/9)):
	output=output.rename(index={i:'id_'+str(i)})

output.to_csv(outputfilename)