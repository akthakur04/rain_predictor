import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

url = "rainfall in india 1901-2015.csv"
data=pd.read_csv(url)
#print("DATA HEADS : ")
#print(data.info())
print(data.shape)
print (data.isnull().sum())


#replace nan data with mean of that column


mean=data.mean()
data=data.replace(np.nan,mean)
print((data.isnull().sum()/len(data))*100)
print(data.head(40))


#print(data.groupby('SUBDIVISION').size())
#print("covarience: ", data.cov())
#print("corelation:", data.corr())

#corr_cols=data.corr()['ANNUAL'].sort_values()[::-1]
#print("index of corr cols : ",corr_cols.index)


#visualise
#print("scatter plot ")
#plt.scatter(data.ANNUAL,data.JAN)
#sns.regplot(x="JAN",y="ANNUAL",data=data)
#plt.show()



#trainning and testing data
#70% train
y=data['ANNUAL']
X=data[['Jan-Feb','Mar-May','Jun-Sep',"Oct-Dec"]]
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.3,shuffle=False)

print("train X shape",train_x.shape,"; test X shape",test_x.shape)
print("train Y shape",train_y.shape,"; test X shape",test_y.shape)
lr=LinearRegression()
lr.fit(train_x,train_y)
pred=lr.predict(test_x)

#print("PREDICTED",pred.flatten())
#print("ACTUAL",test_y)

df=pd.DataFrame({'actual ':test_y,"predicted" :pred})
print(df)


#errors
print("MEAN ABSOLUTE ERROR: ",mean_absolute_error(test_y,pred))
print("MEAN SQ ERROR: ", mean_squared_error(test_y,pred))
print("Root mean square error", np.sqrt(mean_squared_error(test_y,pred)))


#regression plot
sns.regplot(x=pred,y=test_y)
plt.show()




#80% train
y=data['ANNUAL']
X=data[['Jan-Feb','Mar-May','Jun-Sep',"Oct-Dec"]]
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,shuffle=False)

print("train X shape",train_x.shape,"; test X shape",test_x.shape)
print("train Y shape",train_y.shape,"; test X shape",test_y.shape)
lr=LinearRegression()
lr.fit(train_x,train_y)
pred=lr.predict(test_x)

#print("PREDICTED",pred.flatten())
#print("ACTUAL",test_y)

df=pd.DataFrame({'actual ':test_y,"predicted" :pred})
print(df)

#errors
print("MEAN ABSOLUTE ERROR: ",mean_absolute_error(test_y,pred))
print("MEAN SQ ERROR: ", mean_squared_error(test_y,pred))
print("Root mean square error", np.sqrt(mean_squared_error(test_y,pred)))


#regression plot
sns.regplot(x=pred,y=test_y)
plt.show()

#90% train
y=data['ANNUAL']
X=data[['Jan-Feb','Mar-May','Jun-Sep',"Oct-Dec"]]
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.1,shuffle=False)

print("train X shape",train_x.shape,"; test X shape",test_x.shape)
print("train Y shape",train_y.shape,"; test X shape",test_y.shape)
lr=LinearRegression()
lr.fit(train_x,train_y)
pred=lr.predict(test_x)

#print("PREDICTED",pred.flatten())
#print("ACTUAL",test_y)
df=pd.DataFrame({'actual ':test_y,"predicted" :pred})
print(df)


#errors
print("MEAN ABSOLUTE ERROR: ",mean_absolute_error(test_y,pred))
print("MEAN SQ ERROR: ", mean_squared_error(test_y,pred))
print("Root mean square error", np.sqrt(mean_squared_error(test_y,pred)))


#regression plot
sns.regplot(x=pred,y=test_y)
plt.show()

