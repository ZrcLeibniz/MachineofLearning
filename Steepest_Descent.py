import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 载入数据
# 可以使用np包中的genfromtxt方法来导入数据，其中delimiter属性用来表示文件的分隔符
data = np.genfromtxt('data.csv', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]
# plt.figure(figsize=(10, 10), dpi=80)
# plt.scatter(x_data, y_data, color='red')
# plt.show()

# 学习率
lr = 0.0001
# 截距
b = 0
# 斜率
k = 0
# 最大迭代次数
epochs = 50


# 定义出代价函数
def compute_error(b, k, x_data, y_data, ):
    totalError = 0
    for i in range(len(x_data)):
        totalError += ((k * x_data[i] + b) - y_data[i]) ** 2
    return totalError / float(len(x_data)) / 2.0


# 求出b k两个参数
def gradient_decent_runner(x_data, y_data, b, k, lr, epochs):
    # 计算总数据量
    m = float(len(x_data))
    # 梯度下降迭代epochs次
    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        for j in range(len(x_data)):
            b_grad += (1 / m) * ((k * x_data[i] + b) - y_data[i])
            k_grad += (1 / m) * (x_data[i]) * ((k * x_data[i] + b) - y_data[i])
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)
        # # 每迭代5次画一次图
        # if i % 5 == 0:
        #     print('epochs:', i)
        #     plt.title('线性回归-梯度下降')
        #     plt.plot(x_data, y_data, 'b.')
        #     plt.plot(x_data, k * x_data + b, 'r-')
        #     plt.figure(figsize=(10, 10), dpi=80)
        #     plt.show()

    return b, k


print('Starting b = {0}, k = {1}, error = {2}'.format(b, k, compute_error(b, k, x_data, y_data)))
print('Running')
b, k = gradient_decent_runner(x_data, y_data, k, b, lr, epochs)
print('After b = {0}, k = {1}, error = {2}'.format(b, k, compute_error(b, k, x_data, y_data)))

# 画图
plt.title('线性回归-梯度下降')
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k * x_data + b, 'r-')
plt.figure(figsize=(10, 10), dpi=80)
plt.show()
