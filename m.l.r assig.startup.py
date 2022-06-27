
import pandas as pd

df = pd.read_csv('C:\\python notes\\ASSIGNMENTS\\m.l.r assig\\50_Startups.csv')
df.shape
df.head()
df.info() # 'state' column is in object
df.corr()

'''
                 R&D Spend  Administration  Marketing Spend    Marketing Spend
R&D Spend         1.000000        0.241955         0.724248  0.972900
Administration    0.241955        1.000000        -0.032154  0.200717
Marketing Spend   0.724248       -0.032154         1.000000  0.747766
Profit            0.972900        0.200717         0.747766  1.000000
'''
# label encoding (convert categorical to numeric)

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['State'] = LE.fit_transform(df['State'])
df['State'].value_counts()
df

# split x,y variables

X=df.drop('Profit',axis=1)
X

y=df[['Profit']]
y

# by using train-test, spliting the varaibles by 70:30
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


#regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

#for predict the test values
y_prdict=reg.predict(X_test)
y_prdict


# TRAIN-TEST SCORE
test_data_model_score=reg.score(X_test,y_test)
print ('score of test data',test_data_model_score)
# 0.9355139722149947

train_data_model_score=reg.score(X_train,y_train)
print ('score of test data',train_data_model_score)
# 0.9515496105627431

# FOR M.A.E
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test, y_prdict)
mae
#6503.577323580032

# FOR M.S.E
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_prdict)
mse
# 62244962.38946449

import numpy as np

# FOR R.M.S.E
from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test, y_prdict))
rmse
# 7889.547666974609

# FOR M.S.L.E
from sklearn.metrics import mean_squared_log_error
rmsle=np.sqrt(mean_squared_log_error(y_test, y_prdict))
rmsle
# 0.07707849995361643


# FOR R2 SCORE
from sklearn.metrics import r2_score
r2score= r2_score(y_prdict,y_test)*100
r2score
#93.39448007716635

# SCATTER PLOT 
import matplotlib.pyplot as plt

plt.scatter(y_test["Profit"], y_prdict)
plt.show()

#--------------------------------------------------------------------









































