

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression

data =pd.read_csv('Advertising.csv',usecols=['TV','radio','newspaper','sales'])
df=pd.DataFrame(data)

############################################## understanding data
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.describe())
# print(df.info)
# print(df.isnull().sum())
# missingno.matrix(df)
# plt.show()
##############################################
df["benefit-TV"] = 150-df['TV']
# print(df.head(10).to_string())
# print(df.corr())

from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler

le=LabelEncoder()
df['benefil1'] = le.fit_transform(df['benefit-TV'])
# print(df.columns)


X = df.drop('sales',axis=1)
Y = df['sales']
# print(X.head(10).to_string())



ss =StandardScaler()
ss_scal=ss.fit_transform(X)
# print(ss_scal)

X_train, X_test, Y_train, Y_test= train_test_split(ss_scal,Y ,test_size=0.20 ,random_state=0 )


lr=LinearRegression()
lr.fit(X_train , Y_train)
y_pred=lr.predict(X_test)
print(y_pred)


print('mae :', mean_absolute_error(Y_test,y_pred))
print('mse :', mean_squared_error(Y_test,y_pred))
print('rmse :', np.sqrt(mean_squared_error(Y_test,y_pred)))
print('R2score :', r2_score(Y_test,y_pred)*100)


