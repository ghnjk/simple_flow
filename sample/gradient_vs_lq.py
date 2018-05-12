#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import matplotlib.pyplot as plt
import random
import numpy as np


def show_line(a, b, label, color):
    x = np.linspace(-1, 1, 200)[:, np.newaxis]
    y = [a * i + b for i in x]
    plt.plot(x, y, linewidth=3.0, label=label, color=color)


def calc_average_loss(x_data, y_data, a, b):
    sum_loss = 0
    for x, y in zip(x_data, y_data):
        y_p = x * a + b
        sum_loss += np.sum(np.power(y_p - y, 2))
    return sum_loss / float(len(x_data))


def predict_with_gradient(x_data, y_data):
    from simple_flow.flow import PlaceHolder, Variable, add_flow
    from simple_flow.nn import ConstantInitializer, Add, Pow, ReduceMean, Multify, Sub
    from simple_flow.model import Model
    from simple_flow.train import GradientDecent
    batch_size = 16
    x = PlaceHolder(name="x", shape=[None, 1])
    y = PlaceHolder(name="y", shape=[None, 1])
    a = Variable(name="a", shape=[1, ])
    b = Variable(name="b", shape=[1, ], initializer=ConstantInitializer(0.1))
    ax = add_flow(x, Multify(a), name="a * x")
    y_predict = add_flow(ax, Add(b), name="a * x + b")
    loss = add_flow(y_predict, Sub(y), name="y_predict - y")
    loss = add_flow(loss, Pow(n=2), name="(y_predict - y)^2")
    loss = add_flow(loss, ReduceMean(), name="loss")
    model = Model(losses=loss, predicts=y_predict, optimizer=GradientDecent())
    for epoch in range(200):
        batch = np.array(random.sample(zip(x_data, y_data), batch_size))
        x_batch = batch[:, 0]
        y_batch = batch[:, 1]
        feed_dict = {
            x: x_batch,
            y: y_batch
        }
        model.fit(feed_dict=feed_dict, max_train_itr=50, verbose=0)
    print("gradient decent: a * x + b: a = ", a.values[0], "b = ", b.values[0],
          "loss: ", loss.values)
    avg_loss = calc_average_loss(x_data, y_data, a.values, b.values)
    print("gradient decent average loss: ", avg_loss)
    show_line(a.values, b.values, "gradient decent", color="red")


def hypfunc(x, a):
    # 输入：x 横坐标数值， A 多项式系数 [a0,a1,...,an-1]
    # 返回 y = hypfunc(x)
    return np.sum(a[i]*(x**i) for i in range(len(a)))


# 使用 θ = (X.T*X + λI)^-1 * X.T * y求解直线参数
# 该函数会在X的前面添加偏移位X0 = 1
def predict_with_least_square(x_data, y_data, lam=0.01):
    x = np.array(x_data).reshape(x_data.shape[0])
    y = np.array(y_data).reshape(y_data.shape[0])
    x = np.vstack((np.ones((len(x),)), x))  # 往上面添加X0
    x = np.mat(x).T     # (m,n)
    y = np.mat(y).T     # (m,1)
    m, n = x.shape
    eye = np.eye(n, n)    # 单位矩阵

    theta = ((x.T * x + lam * eye) ** -1) * x.T * y  # 核心公式
    theta = np.array(np.reshape(theta, len(theta)))[0]
    b = theta[0]
    a = theta[1]
    print("least_square a: ", a, "b: ", b)
    avg_loss = calc_average_loss(x_data, y_data, a, b)
    print("least_square average loss: ", avg_loss)
    show_line(a, b, "least_square", color="yellow")


def main():
    x_data = np.linspace(-1, 1, 200)[:, np.newaxis]
    noise = np.random.normal(0, 2, x_data.shape)
    a = -50.3
    b = 309.33
    y_data = [a * i + b for i in x_data] + noise
    show_line(a, b + np.mean(noise), "real", color="blue")
    plt.scatter(x_data, y_data)
    predict_with_gradient(x_data, y_data)
    predict_with_least_square(x_data, y_data)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    sys.path.append("..")
    main()
