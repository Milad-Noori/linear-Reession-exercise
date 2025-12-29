import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno

data =pd.read_csv('Advertising.csv',usecols=['TV','radio','newspaper','sales'])
df=pd.DataFrame(data)

############################################## understanding data
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.describe())
# print(df.info)
print(df.isnull().sum())
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
print(X.head(10).to_string())






