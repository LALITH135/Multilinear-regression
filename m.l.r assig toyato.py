

# multi linear regression

import pandas as pd

df = pd.read_csv('C:\\python notes\\ASSIGNMENTS\\m.l.r assig\\ToyotaCorolla.csv',encoding='latin1')
df.shape
df.head()
list(df) 
df.corr()
df.describe()

# preparing a prediction model for the below columns,by 'CORRELATION' & AVOID THE WEEK RELATION BETWEEN X VARIABLES.

data=[("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
list(data)

# split the variables
Y = df["Price"]

# THE BELOW X VARIABLES ARE UNDER COMMENT,WE CAN CHECK THE EACH BY REMOVING THE # TAG

X = df["Age_08_04"]  
# X = df[["Age_08_04","Weight"]]
# X = df[["Age_08_04","Weight","KM"]]
# X = df[["Age_08_04","Weight","KM","HP"]]
# X = df[["Age_08_04","Weight","KM","HP","Quarterly_Tax"]]


import numpy as np
X = X[:, np.newaxis] # converting in to 2 D arrary from 1 D array

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,Y)
Y_pred = lm.predict(X)

# FOR R.M.S.E
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
RMSE = np.sqrt(mse)
print("RMSE:" , RMSE.round(4))

# FOR M.S.E
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean squared error: ", mse.round(2))

#FOR R2 ADJUSTED

def adj_r2(X,y,y_pred):
    y_mean = np.mean(y)
    num = np.sum((y - y_pred) ** 2) / (len(y) - ((X.shape[1])+1))
    den = np.sum((y-y_mean) ** 2) / (len(y)-1)
    adjusteddR2 = 1 - (num/den)
    print("R2 adjusted:" , (adjusteddR2*100).round(4))

adj_r2(X,Y,Y_pred)

# SCATTER PLOT BY EACH X&Y

df.plot.scatter(x = 'Age_08_04',y = 'Price')
df.plot.scatter(x = 'KM',y ='Price')
df.plot.scatter(x = 'cc',y = 'Price')
df.plot.scatter(x = 'Gears',y ='Price')
df.plot.scatter(x = 'HP',y = 'Price')
df.plot.scatter(x = 'Doors',y = 'Price')
df.plot.scatter(x = 'Quarterly_Tax',y = 'Price')
df.plot.scatter(x = 'Weight',y = 'Price')


#----------------------------------------
# "Age_08_04":
# Mse :  3044403.46, R2 adjusted: 76.8249, RMSE: 1744.8219

# "Age_08_04","Weight":
#  Mse :  2562472.82, R2 adjusted: 80.48, RMSE: 1600.7726

# "Age_08_04","Weight","KM"
# Mse:  1996777.76,  R2 adjusted: 84.7786, RMSE: 1413.0739

# "Age_08_04","Weight","KM","HP"
# Mse:  1817054.53,  R2 adjusted: 86.1389, RMSE: 1347.9816

#"Age_08_04","Weight","KM","HP","Quarterly_Tax"
# Mse:  1805234.88,  R2 adjusted: 86.2195, RMSE: 1343.5903

