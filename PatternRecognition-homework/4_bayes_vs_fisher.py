#!usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange
from numpy import meshgrid

plt.axis([0, 20, 0, 20])
plt.xlabel('length')
plt.ylabel('light')

data_1 = 2 * np.random.randn(20, 2) + 4   # 鲈鱼   均值为2,方差为4的正态分布
data_2 = 4 * np.random.randn(20, 2) + 10  # 鲑鱼

# data_test = np.random.random_integers(0, 20, size=(1, 2))
data_test = np.random.uniform(0, 20, size=(1, 2))   # 测试样本  (1, 2)

for i in range(0, 20):
    plt.plot(data_1[i, 0], data_1[i, 1], 'ro')
for i in range(0, 20):
    plt.plot(data_2[i, 0], data_2[i, 1], 'go')
plt.plot(data_test[0, 0], data_test[0, 1], 'ys')


def fisher():
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

    # g(x) = Wopt.T * x + b
    test = Wopt.T * data_test.T  # (1,1)
    if test - b > 0:
        output = 'fisher判别结果: 鲈鱼'
    else:
        output = 'fisher判别结果: 鲑鱼'

    # 画出分类面
    w1 = Wopt[0, 0]
    w2 = Wopt[1, 0]
    x = [i + 1 for i in range(20)]  # [1 2 ... 20]
    x = np.array(x)       # (20,)
    # w.T * x + b = 0    (w.T = (w1, w2))
    # 转化为: w1 * x + w2 * y + b = 0   从而:
    y = - w1 * x / w2 + b / w2
    y = np.array(y)       # (1, 20)
    plt.plot(x, y[0], 'm')     # 洋红色为fisher分类面

    # 计算判别准确率
    d1 = Wopt.T * data_1.T     # (1,20)
    d2 = Wopt.T * data_2.T

    right_count, wrong_count = 0, 0
    for i in range(0, 20):
        if d1[0, i] - b > 0:
            right_count += 1
        else:
            wrong_count += 1

        if d2[0, i] - b > 0:
            wrong_count += 1
        else:
            right_count += 1
    accuracy = right_count / N

    plt.plot(m1[0], m1[1], 'ko')   # 原始数据的均值向量
    plt.plot(m2[0], m2[1], 'bo')

    return output, wrong_count, accuracy


def bayes():
    # 计算两个类别的均值、协方差
    u1 = np.mat(np.mean(data_1, axis=0)).T  # (2, 1)
    u2 = np.mat(np.mean(data_2, axis=0)).T
    sigma1 = np.mat(np.cov(data_1.T))  # (2, 2)
    sigma2 = np.mat(np.cov(data_2.T))
    p1 = p2 = 0.5  # 指定先验概率

    right_count, wrong_count = 0, 0
    if sigma1 is sigma2:
        w = sigma1.I * (u1 - u2)  # (2, 1)
        x0 = (u1 + u2) / 2 - np.log(p1 / p2) * (u1 - u2) / ((u1 - u2).T * sigma1.I * (u1 - u2))  # (2, 1)

        x = [i + 1 for i in range(20)]  # [1 2 ... 20]
        x = np.array(x)
        # 两类协方差相同的情况下的分类边界为： w.T * (x - x0) = 0    (设 w.T = (w1, w2))
        # 转化为: w1 * (x - x0) + w2 * (y - y0) = 0   从而:
        y = x0[1, 0] - w[0, 0] * (x - x0[0, 0]) / w[1, 0]
        plt.plot(x, y)

        if w[0, 0] * (data_test[0, 1] - x0[1, 0]) + w[1, 0] * (data_test[0, 0] - x0[0, 0]) < 0:
            output = 'bayes判别结果: 鲑鱼'
        else:
            output = 'bayes判别结果: 鲈鱼'

        # 计算判别准确率
        for i in range(0, 20):
            if w[0, 0] * (data_1[i, 1] - x0[1, 0]) + w[1, 0] * (data_1[i, 0] - x0[0, 0]) > 0:
                right_count += 1
            else:
                wrong_count += 1

            if w[0, 0] * (data_2[i, 1] - x0[1, 0]) + w[1, 0] * (data_2[i, 0] - x0[0, 0]) > 0:
                wrong_count += 1
            else:
                right_count += 1

    else:
        W1 = - sigma1.I / 2
        W2 = - sigma2.I / 2
        W = W1 - W2  # (2, 2)

        w1 = sigma1.I * u1
        w2 = sigma2.I * u2
        w = w1 - w2  # (2, 1)

        w10 = - u1.T * sigma1.I * u1 / 2 - np.log(np.linalg.det(sigma1)) / 2 + np.log(p1)
        w20 = - u2.T * sigma2.I * u2 / 2 - np.log(np.linalg.det(sigma2)) / 2 + np.log(p2)
        w0 = w10 - w20  # (1, 1)

        E1 = W[0, 0]
        E23 = W[0, 1] + W[1, 0]
        E4 = W[1, 1]

        e1 = w[0, 0]
        e2 = w[1, 0]

        delta = 0.025
        xrange = arange(-5.0, 20.0, delta)
        yrange = arange(-5.0, 20.0, delta)
        x, y = meshgrid(xrange, yrange)

        F = E1 * x * x + E23 * x * y + E4 * y * y + e1 * x + e2 * y + w0
        plt.contour(x, y, F, [1])

        if (E1 * data_test[0, 0] * data_test[0, 0] + E23 * data_test[0, 0] * data_test[0, 1] + E4 * data_test[0, 1]
            * data_test[0, 1] + e1 * data_test[0, 0] + e2 * data_test[0, 1] + w0 > 0):
            output = 'bayes判别结果: 鲈鱼'
        else:
            output = 'bayes判别结果: 鲑鱼'

        # 计算判别准确率
        for i in range(0, 20):
            if E1 * data_1[i, 0] * data_1[i, 0] + E23 * data_1[i, 0] * data_1[i, 1] + E4 * data_1[i, 1] \
                    * data_1[i, 1] + e1 * data_1[i, 0] + e2 * data_1[i, 1] + w0 > 0:
                right_count += 1
            else:
                wrong_count += 1

            if E1 * data_2[i, 0] * data_2[i, 0] + E23 * data_2[i, 0] * data_2[i, 1] + E4 * data_2[i, 1] \
                    * data_2[i, 1] + e1 * data_2[i, 0] + e2 * data_2[i, 1] + w0 > 0:
                wrong_count += 1
            else:
                right_count += 1

    accuracy = right_count / N
    return output, wrong_count, accuracy


if __name__ == '__main__':
    N = 40
    print('对于随机测试样本(%f, %f):' % (data_test[0][0], data_test[0][1]))
    output_f, wrong_f, accuracy_f = fisher()
    output_b, wrong_b, accuracy_b = bayes()
    print(output_f)
    print(output_b)
    print('\nfisher和bayes比较:')
    print('fisher错分%d个, 判别准确率为: %f\nbayes错分%d个, 判别准确率为: %f' % (wrong_f, accuracy_f, wrong_b, accuracy_b))
    plt.show()
