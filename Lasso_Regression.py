import numpy as np 
from sklearn import linear_model

data = np.genfromtxt('longley.csv', delimiter=',')
x_data = data[1:, 2:]
y_data = data[1:, 1]
model = linear_model.LassoCV()
model.fit(x_data, y_data)
# lasso相关系数
print(model.alpha_)
# 相关系数
print(model.coef_)
model.predict(x_data[-2, np.newaxis])
