import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import PolynomialFeatures

# 生成二维正态分布，生成的数据分为两类，500个样本，2个样本特征
x_data, y_data = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)



# logistic = linear_model.LogisticRegression()
# logistic.fit(x_data, y_data)
#
# x_min, x_max = x_data[:, 0].min()-1, x_data[:, 0].max()+1
# y_min, y_max = x_data[:, 1].min()-1, x_data[:, 1].max()+1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
# z = logistic.predict(np.c_[xx.ravel(), yy.ravel()])
# z = z.reshape(xx.shape)
# cs = plt.contourf(xx, yy, z)
# plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
# plt.show()
# print('score', logistic.score(x_data, y_data))


# 定义多项式回归
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x_data)
logistic = linear_model.LogisticRegression()
logistic.fit(x_poly, y_data)

x_min, x_max = x_data[:, 0].min()-1, x_data[:, 0].max()+1
y_min, y_max = x_data[:, 1].min()-1, x_data[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
z = logistic.predict(poly_reg.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
z = z.reshape(xx.shape)
cs = plt.contourf(xx, yy, z)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
print('score', logistic.score(x_poly, y_data))