#!usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.axis([0, 20, 0, 20])
    plt.xlabel('length')
    plt.ylabel('light')

    data_1 = 2 * np.random.randn(20, 2) + 4   # 鲈鱼   均值为2,方差为4的正态分布
    data_2 = 4 * np.random.randn(20, 2) + 10  # 鲑鱼

    data_test = np.random.uniform(0, 20, size=(1, 2))   # 测试样本  (1, 2)
    # data_test = np.random.random_integers(0, 20, size=(1, 2))

    for i in range(0, 20):
        plt.plot(data_1[i, 0], data_1[i, 1], 'ro')    # 红色圆点为鲈鱼
    for i in range(0, 20):
        plt.plot(data_2[i, 0], data_2[i, 1], 'go')    # 绿色圆点为鲑鱼
    plt.plot(data_test[0, 0], data_test[0, 1], 'ys')  # 黄色方块为测试数据

    sigma1 = np.mat(np.cov(data_1.T))   # 鲈鱼协方差  (2, 2)
    sigma2 = np.mat(np.cov(data_2.T))   # 鲑鱼协方差
    m1 = np.mat(np.mean(data_1, axis=0)).T    # 原始数据的均值向量    (2, 1)
    m2 = np.mat(np.mean(data_2, axis=0)).T
    S1 = sigma1 * (20 - 1)       # 原始数据的离散度矩阵  (2, 2)
    S2 = sigma2 * (20 - 1)

    # Fisher线性分类
    Wopt = (S1 + S2).I * (m1 - m2)    # (2, 1)
    u1 = Wopt.T * m1                  # 投影后一维数据的均值
    u2 = Wopt.T * m2
    b = (u1 + u2) / 2

    test = Wopt.T * data_test.T    # (1,1)

    # gx = Wopt.T * x + b
    if test - b > 0:
        print('测试样本(%f, %f)为鲈鱼' % (data_test[0, 0], data_test[0, 1]))
    else:
        print('测试样本(%f, %f)为鲑鱼' % (data_test[0, 0], data_test[0, 1]))

    plt.plot(m1[0], m1[1], 'ko')   # 原始数据的均值向量# 黑色圆点为鲈鱼原始均值
    plt.plot(m2[0], m2[1], 'bo')   # 蓝色圆点为鲑鱼原始均值

    # 画出分类面
    w1 = Wopt[0, 0]
    w2 = Wopt[1, 0]
    # print(w1)
    x = [i + 1 for i in range(20)]  # [1 2 ... 20]
    x = np.array(x)   # (20,)
    # w.T * x + b = 0    (设 w.T = (w1, w2))
    # 转化为: w1 * x + w2 * y + b = 0   从而:
    y = - w1 * x / w2 + b / w2
    y = np.array(y)
    # print(y.shape)     # (1, 20)
    plt.plot(x, y[0])
    plt.show()
