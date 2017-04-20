#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
有两类样本（如鲈鱼和鲑鱼），每个样本有两个特征（如长度和亮度），每类有若干个（比如20个）样本点，假设每类样本点服从二维正态分布，
自己随机给出具体数据，计算每类数据的均值点，并且把两个均值点连成一线段，用垂直平分该线段的直线作为分类边界。
再根据该分类边界对一随机给出的样本判别类别。画出图形。
"""
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.axis([0, 20, 0, 20])
    plt.xlabel('length')
    plt.ylabel('light')

    data_1 = 2 * np.random.randn(20, 2) + 4   # 鲈鱼   均值为2,方差为4的正态分布
    data_2 = 4 * np.random.randn(20, 2) + 10  # 鲑鱼
    data_test = np.random.uniform(0, 20, size=(1, 2))

    for i in range(0, 20):
        plt.plot(data_1[i, 0], data_1[i, 1], 'ro')
    for i in range(0, 20):
        plt.plot(data_2[i, 0], data_2[i, 1], 'go')

    plt.plot(data_test[0, 0], data_test[0, 1], 'ys')

    u1 = np.mean(data_1, axis=0)   # [ 4.20674584  3.57477034]
    u2 = np.mean(data_2, axis=0)

    k = -1 / ((u1[1] - u2[1]) / (u1[0] - u2[1]))
    y = (u1[1] + u2[1]) / 2
    x = (u1[0] + u2[0]) / 2
    b = y - k * x

    def classify(m, n):
        if k * m + b - n > 0: return 'down'
        return 'up'

    if classify(data_test[0][0], data_test[0][1]) is classify(u1[0], u1[1]):
        print('测试样本(%f, %f)为鲈鱼' % (data_test[0, 0], data_test[0, 1]))
    else:
        print('测试样本(%f, %f)为鲑鱼' % (data_test[0, 0], data_test[0, 1]))

    plot_dot1 = [-b / k, 0]
    plot_dot2 = [0, b]
    plt.plot(plot_dot1, plot_dot2)
    plt.show()

