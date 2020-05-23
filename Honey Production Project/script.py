import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

prod_per_year = df.groupby('year').totalprod.mean().reset_index()

X = prod_per_year['year']
X = X.values.reshape(-1, 1)

y = prod_per_year['totalprod']
y = y.values.reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(X, y)
#print(regr.coef_[0],regr.intercept_)
y_predict = regr.predict(X)

X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)

future_predict = regr.predict(X_future)

plt.plot(X, y_predict)
plt.scatter(X, y)
plt.show()
plt.plot(X_future, future_predict)
plt.show()

last = future_predict[-1]
print(last)

