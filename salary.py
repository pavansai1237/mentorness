import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
data = pd.read_csv('Salary_Data.csv')
data.head()
data.shape
data.describe()
sns.pairplot(data, y_vars=['SALARY'], x_vars=['PAST EXP'])
data['FIRST NAME'] = pd.to_numeric(data['FIRST NAME'], errors='coerce')
X = data['PAST EXP']
y = data['SALARY']
X_train,X_test,y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 100)
X_train.shape
X_test.shape
X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()
print(model.summary())
plt.scatter(X_train,y_train)
plt.plot(X_train, 25200 + X_train * 9731.2038,'r')
plt.show()
y_train_pred = model.predict(X_train_sm)
y_train_pred.head()
residual = (y_train - y_train_pred)
residual.head()
sns.distplot(residual)
sns.scatterplot(X_train)
X_test_sm = sm.add_constant(X_test)
y_pred = model.predict(X_test_sm)
RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
RMSE
r2_score(y_test,y_pred)
plt.scatter(X_test,y_test)
plt.plot(X_test, 25200 + X_test * 9731.2038,'r')
plt.show()