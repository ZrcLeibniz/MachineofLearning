import copy
import numpy as np
# 使用梯度下降算法来实现宝可梦进化后战斗力的预测
# 将精灵宝可梦的数据输入本函数，已经知道的是宝可梦的预测模型
# 需要传入的形参有宝可梦的各个特征值，以及使用此方法时想要设置的学习率
# 从自己设计的简化模型的角度来看宝可梦进化后的战斗力与宝可梦的种族、宝可梦的身高，宝可梦的体重，以及宝可梦
# 进化前的战斗力有关
# 这里假设宝可梦的种族只有4种分别是：Pidegey、Weddle、Caterpie、Eevee
# 这里的第一步可以将宝可梦进行分类，例如如果是Pidegey就将其放入Pidegey的数据集之中。
# 经过思考和实践之后发现，没有必要实现分类，可以可以通过一个函数将输入的数据集的种类变换成1或者0
# 原因是因为在模型中并没有将精灵宝可梦的种族进行区分，而是通过一个Delta函数将宝可梦的数据有效或者无效化
# 之所以要进行分类是因为通过对不同种族宝可梦的预测可以增加模型准确性，可以对不同种族的宝可梦使用不同的
# 回归线来进行表示
# 所以这里要做的第一步就是写出一种函数将宝可梦的种族有效化或者无效化
# 之后要考虑的就是对模型的参数进行调整，那么第一步应该构建出精灵宝可梦战斗力预测的模型
# data_model = ['种族', '进化前的战斗力', '进化前战斗力的平
# 方', '身高', '体重', '进化后的战斗力']
# x_data_model = data_model[0:4:1]
# y_data_model = data_model[-1]
# print(x_data_model)
# x_data1 = [1, 2, 3]
# y_data1 = [4, 5, 6]
Test_Pokemon = [
    ['Pidegey', 100, 98, 9604, 120, 130],
    ['Weddle', 87, 65, 4225, 12, 89],
    ['Caterpie', 76, 55, 3025, 56, 99],
    ['Eevee', 130, 34, 1156, 55, 200],
    ['Pidegey', 100, 98, 9604, 120, 130],
    ['Weddle', 87, 65, 9604, 12, 89],
    ['Caterpie', 76, 55, 3025, 56, 99],
    ['Eevee', 130, 34, 1156, 55, 200]]


def delta_Pidegey(Pokemon_data):
    Pokemon_data_up = copy.deepcopy(Pokemon_data)
    for i in range(len(Pokemon_data)):
        if Pokemon_data[i][0] == 'Pidegey':
            Pokemon_data_up[i][0] = 1
        else:
            Pokemon_data_up[i][0] = 0
    return Pokemon_data_up[:0]


# 如果宝可梦的种族是Weddle那么就让它的种族显示为1，否则显示为0
def delta_Weddle(Pokemon_data):
    Pokemon_data_up = copy.deepcopy(Pokemon_data)
    for i in range(len(Pokemon_data)):
        if Pokemon_data[i][0] == 'Weddle':
            Pokemon_data_up[i][0] = 1
        else:
            Pokemon_data_up[i][0] = 0
    return Pokemon_data_up[:0]


# 如果宝可梦的种族是Pokemon那么就让它的种族显示为1，否则显示为0
def delta_Caterpie(Pokemon_data):
    Pokemon_data_up = copy.deepcopy(Pokemon_data)
    for i in range(len(Pokemon_data)):
        if Pokemon_data[i][0] == 'Caterpie':
            Pokemon_data_up[i][0] = 1
        else:
            Pokemon_data_up[i][0] = 0
    return Pokemon_data_up[:0]


# 如果宝可梦的种族是Eevee那么就让它的种族显示为1，否则显示为0
def delta_Eevee(Pokemon_data):
    Pokemon_data_up = copy.deepcopy(Pokemon_data)
    for i in range(len(Pokemon_data)):
        if Pokemon_data[i][0] == 'Eevee':
            Pokemon_data_up[i][0] = 1
        else:
            Pokemon_data_up[i][0] = 0
    return Pokemon_data_up[:0]


Test_Pidegey_data = delta_Pidegey(Test_Pokemon)
Test_Weddle_data = delta_Weddle(Test_Pokemon)
Test_Caterpie_data = delta_Caterpie(Test_Pokemon)
Test_Eevee_data = delta_Eevee(Test_Pokemon)
print('weddle', Test_Weddle_data)
print('pidegy', Test_Pidegey_data)
print('caterpie', Test_Caterpie_data)
print('eevee', Test_Eevee_data)
a = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
print(len(a))

# 由于精灵宝可梦的战斗力在预测过程中如果只考虑未进行升级时的战斗力的话，会使模型一直都
# 处于一个线性状态，而因为线性模型的表示能力有很大的局限性，所以加入了升级以前战斗力的
# 平方，这个时候就必须保证传入的数据足够规范。加入平方后会引入非线性项，使得模型的表达能
# 力得到提升。
# def Pokemon_CP_Model(Pokemon_data):
#     delta_Pidegey(Pokemon_data)



# print(Test_Pokemon[1])
# Pidegey_data = []
# Weddle_data = []
# Caterpie_data = []
# Eevee_data = []
# for j in range(len(Test_Pokemon)):
#     if Test_Pokemon[j][0] == 'Pidegey':
#         Pidegey_data.append(Test_Pokemon[j])
#     elif Test_Pokemon[j][0] == ' Weddle':
#         Weddle_data.append(Test_Pokemon[[j]])
#     elif Test_Pokemon[j][0] == 'Caterpie':
#         Caterpie_data.append(Test_Pokemon[j])
#     elif Test_Pokemon[j][0] == ' Eevee':
#         Eevee_data.append(Test_Pokemon[j])
#
# print(Pidegey_data)
# print(Weddle_data)
# print(Eevee_data)
# global Pidegey_data, Weddle_data, Caterpie_data, Eevee_data


# def Pokemon_Classification(x_Pokemon):
#     Pidegey_data = []
#     Weddle_data = []
#     Caterpie_data = []
#     Eevee_data = []
#     for j in range(len(x_Pokemon)):
#         if x_Pokemon[j][0] == 'Pidegey':
#             Pidegey_data.append(x_Pokemon[j])
#     for n in range(len(x_Pokemon)):
#         if x_Pokemon[n][0] == ' Weddle':
#             Weddle_data.append(x_Pokemon[[n]])
#     for o in range(len(x_Pokemon)):
#         if x_Pokemon[o][0] == 'Caterpie':
#             Caterpie_data.append(x_Pokemon[o])
#     for p in range(len(x_Pokemon)):
#         if x_Pokemon[p][0] == 'Eevee':
#             Eevee_data.append(x_Pokemon[p])
#         print(Weddle_data) #Pidegey_data, Weddle_data, Caterpie_data, Eevee_data

# Pokemon_Classification(Test_Pokemon)
# Pidegey_data, Weddle_data, Caterpie_data, Eevee_data = Pokemon_Classification(Test_Pokemon)
# print(Pidegey_data)
# print(Weddle_data)
# print(Caterpie_data)
# print(Eevee_data)

# def Pokemon_Predict(learn, x_data=[], y_data=[]):
#     print(x_data)
#     print(y_data)
#
#
# Pokemon(x_data1, y_data1)
