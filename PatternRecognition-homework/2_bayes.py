#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
根据贝叶斯公式，给出在 类条件概率密度为正态分布时具体的 判别函数表达式，用此判别函数设计分类器。
数据随机生成，比如生成两类样本（如鲈鱼和鲑鱼），每个样本有两个特征（如长度和亮度），每类有若干个（比如20个）样本点，
假设每类样本点服从二维正态分布，随机生成具体数据，然后估计每类的均值与协方差，
在两类协方差相同的情况下求出分类边界。先验概率自己给定，比如都为0.5。
如果可能，画出在两类协方差不相同的情况下的分类边界。画出图形。
"""
import numpy as np
import matplotlib.pyplot as plt
# from sympy.parsing.sympy_parser import parse_expr
from sympy import *
# from pylab import *     # 引入兼容MATLAB包: pylab
from numpy import arange
from numpy import meshgrid


if __name__ == '__main__':
    plt.axis([0, 20, 0, 20])
    plt.xlabel('length')
    plt.ylabel('light')

    data_1 = 2 * np.random.randn(20, 2) + 4   # 鲈鱼   均值为2,方差为4的正态分布
    data_2 = 4 * np.random.randn(20, 2) + 10  # 鲑鱼
    data_test = np.random.uniform(0, 20, size=(1, 2))  # 测试样本  (1, 2)

    for i in range(0, 20):
        plt.plot(data_1[i, 0], data_1[i, 1], 'ro')
    for i in range(0, 20):
        plt.plot(data_2[i, 0], data_2[i, 1], 'go')

    plt.plot(data_test[0, 0], data_test[0, 1], 'ys')

    # 计算两个类别的均值、协方差
    u1 = np.mat(np.mean(data_1, axis=0)).T  # (2, 1)
    u2 = np.mat(np.mean(data_2, axis=0)).T
    sigma1 = np.mat(np.cov(data_1.T))  # (2, 2)
    sigma2 = np.mat(np.cov(data_2.T))
    p1 = p2 = 0.5  # 指定先验概率

    if sigma1 is sigma2:
        w = sigma1.I * (u1 - u2)  # (2, 1)
        x0 = (u1 + u2) / 2 - np.log(p1 / p2) * (u1 - u2) / ((u1 - u2).T * sigma1.I * (u1 - u2))  # (2, 1)

        x = [i+1 for i in range(20)]  # [1 2 ... 20]
        x = np.array(x)
        # 两类协方差相同的情况下的分类边界为： w.T * (x - x0) = 0    (设 w.T = (w1, w2))
        # 转化为: w1 * (x - x0) + w2 * (y - y0) = 0   从而:
        y = x0[1, 0] - w[0, 0] * (x - x0[0, 0]) / w[1, 0]

        plt.plot(x, y)

        if data_test[0, 1] > y[data_test[0, 0]]:
            print('测试样本(%f, %f)为鲑鱼' % (data_test[0, 0], data_test[0, 1]))
        else:
            print('测试样本(%f, %f)为鲈鱼' % (data_test[0, 0], data_test[0, 1]))

        plt.show()

    else:
        W1 = - sigma1.I / 2
        W2 = - sigma2.I / 2
        W = W1 - W2           # (2, 2)

        w1 = sigma1.I * u1
        w2 = sigma2.I * u2
        w = w1 - w2           # (2, 1)

        w10 = - u1.T * sigma1.I * u1 / 2 - np.log(np.linalg.det(sigma1)) / 2 + np.log(p1)
        w20 = - u2.T * sigma2.I * u2 / 2 - np.log(np.linalg.det(sigma2)) / 2 + np.log(p2)
        w0 = w10 - w20        # (1, 1)

        E1 = W[0, 0]
        E23 = W[0, 1] + W[1, 0]
        E4 = W[1, 1]

        e1 = w[0, 0]
        e2 = w[1, 0]

        if (E1 * data_test[0, 0] * data_test[0, 0] + E23 * data_test[0, 0] * data_test[0, 1] + E4 * data_test[0, 1]
            * data_test[0, 1] + e1 * data_test[0, 0] + e2 * data_test[0, 1] + w0 > 0):
            print('测试样本(%f, %f)为鲈鱼' % (data_test[0, 0], data_test[0, 1]))
        else:
            print('测试样本(%f, %f)为鲑鱼' % (data_test[0, 0], data_test[0, 1]))

        delta = 0.025
        xrange = arange(-5.0, 20.0, delta)
        yrange = arange(-5.0, 20.0, delta)
        x, y = meshgrid(xrange, yrange)

        F = E1 * x * x + E23 * x * y + E4 * y * y + e1 * x + e2 * y + w0
        plt.contour(x, y, F, [1])
        plt.show()

        # 画隐函数图像:
        # ezplot = lambda expr: plot_implicit(parse_expr(expr))
        # x, y = symbols("x y")
        # f = str(E1 * x * x + E23 * x * y + E4 * y * y + e1 * x + e2 * y + w0)
        # ezplot(f)
        # plt.show()

        # f1 = lambdify(x, expre, 'numpy')
        # p1 = f1(a)
        # x = [i + 1 for i in range(20)]  # 1,2,....20
        # x = np.array(x)
        # for i in range(20):
        #     f2 = lambdify(y, p1[i], 'numpy')
        # p2 = f2(a)

        # plt.plot(x, p2)
        # replacements = [(x, y) for i in range(20)]
        # p1 = ezplot(f(a))
        # ezplot(expression.subs(replacements))
        # plt.plot(p1)
