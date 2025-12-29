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
# print(df.isnull().sum())
# missingno.matrix(df)
# plt.show()
##############################################
df["benefit"] = 150-df['TV']
print(df.head(10).to_string())







