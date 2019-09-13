import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# 读入数据
data = genfromtxt('Delivery.csv', delimiter=',')
# print(data)

# 切分数据
x_data = data[:, :-1]
y_data = data[:, -1]

# print(x_data)
# print(y_data)

# 所需参数
theta0 = 0
theta1 = 0
theta2 = 0
lr = 0.0001
epoch = 1000


# 损失函数
def compute_error(theta0, theta1, theta2, x_data, y_data):
    totalError = 0
    for i in range(0, len(y_data)):
        totalError += ((theta1 * x_data[i, 0] + theta2 * x_data[i, 1] + theta0) - y_data[i]) ** 2
    return totalError / float(len(x_data)) / 2


# 求出梯度下降各参数的值
def gradient_descent_runner(theta0, theta1, theta2, x_data, y_data, lr, epoch):
    # 计算总数据量
    m = float(len(x_data))
    # 梯度下降的迭代
    for i in range(epoch):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        for j in range(0, len(x_data)):
            theta0_grad += (1 / m) * (theta1 * x_data[j, 0] + theta2 * x_data[j, 1] - y_data[j])
            theta1_grad += (1 / m) * ((theta1 * x_data[j, 0] + theta2 * x_data[j, 1] + theta0 - y_data[j]) * x_data[j, 0])
            theta2_grad += (1 / m) * ((theta1 * x_data[j, 0] + theta2 * x_data[j, 1] + theta0 - y_data[j]) * x_data[j, 1])
        # 更新位置
        theta0 = theta0 - (lr * theta0_grad)
        theta1 = theta1 - (lr * theta1_grad)
        theta2 = theta2 - (lr * theta2_grad)
    return theta0, theta1, theta2


print("Starting theta0={0}, theta1={1}, theta2={2}, error={3}".format(theta0, theta1, theta2,
                                                                      compute_error(theta0, theta1, theta2, x_data,
                                                                                    y_data)))
print('Running...')
theta0, theta1, theta2 = gradient_descent_runner(theta0, theta1, theta2, x_data, y_data, lr, epoch)
print('After{0} iterations theta0={1}, theta1={2},theta2={3},error={4}'.format(epoch, theta0, theta1, theta2,
                                                                               compute_error(theta0, theta1, theta2,
                                                                                             x_data, y_data)))

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)
x = x_data[:, 0]
y = x_data[:, 1]
# 生成网格矩阵
x, y = np.meshgrid(x, y)
z = theta1 * x + theta2 * y + theta0
# 画3D图
ax.plot_surface(x, y, z)
# 设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')
plt.show()