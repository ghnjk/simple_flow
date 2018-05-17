#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import matplotlib.pyplot as plt
import random
import numpy as np


def show_line(a, b, label, color):
    x = np.linspace(-1, 1, 200)[:, np.newaxis]
    y = [a * i * i + b for i in x]
    plt.plot(x, y, linewidth=3.0, label=label, color=color)


def calc_average_loss(x_data, y_data):
    sum_loss = 0
    for x, y in zip(x_data, y_data):
        y_p = x * a + b
        sum_loss += np.sum(np.power(y_p - y, 2))
    return sum_loss / float(len(x_data))


def min_batch_gradient_decnet(x_data, y_data, a, b):
    from simple_flow.flow import PlaceHolder, Variable, add_flow
    from simple_flow.nn import ConstantInitializer, Add, Pow, ReduceMean, Sub, MatMul, Relu, ReduceSum
    from simple_flow.model import Model
    from simple_flow.train import GradientDecent
    batch_size = 32
    x = PlaceHolder(name="x", shape=[None, 1])
    y = PlaceHolder(name="y", shape=[None, 1])
    w_1 = Variable(name="w_1", shape=[1, 64])
    b_1 = Variable(name="b_2", shape=[64, ], initializer=ConstantInitializer(0.1))
    w_2 = Variable(name="w_2", shape=[64, 1])
    b_2 = Variable(name="b_2", shape=[1, ], initializer=ConstantInitializer(0.1))
    l_1 = add_flow(add_flow(x, MatMul(w_1), "dot(x, w_1)"),
                   Add(b_1),
                   "dot(x, w_1) + b_1")
    l_1 = add_flow(l_1, Relu(), "relu(l_1)")
    y_pre = add_flow(add_flow(l_1, MatMul(w_2), "dot(l_1, w_2)"),
                     Add(b_2),
                     "dot(l_1, w_2) + b_2")
    loss = add_flow(
        add_flow(add_flow(add_flow(y_pre, Sub(y), "y_pre - y"),
                          Pow(n=2),
                          "(y_pre - y)^2"),
                 ReduceSum(axis=1),
                 "sum((y_pre - y)^2)"),
        ReduceMean(),
        "loss"
    )
    model = Model(losses=loss, predicts=y_pre, optimizer=GradientDecent())
    plt.ion()
    for epoch in range(200):
        batch = np.array(random.sample(zip(x_data, y_data), batch_size))
        x_batch = batch[:, 0]
        y_batch = batch[:, 1]
        feed_dict = {
            x: x_batch,
            y: y_batch
        }
        model.fit(feed_dict=feed_dict, max_train_itr=50, verbose=0)
        plt.cla()
        plt.scatter(x_data, y_data)
        show_line(a, b, "real", "blue")
    # print("w_1: ", w_1.values, "b_1: ", b_1.values)
    # print("w_2: ", w_2.values, "b_2: ", b_2.values)
        print("gradient decent average loss: ", loss.values)
        y_draw = model.predict(feed_dict={
            x: x_data
        })
        plt.plot(x_data, y_draw, linewidth=3.0, label="min_batch_gradient_decent", color="red")
        plt.draw()
        plt.pause(0.1)


def main():
    x_data = np.linspace(-1, 1, 200)[:, np.newaxis]
    noise = np.random.normal(0, 2, x_data.shape)
    a = 94
    b = 21
    y_data = [a * i * i + b for i in x_data] + noise
    min_batch_gradient_decnet(x_data, y_data, a, b + np.mean(noise))


if __name__ == '__main__':
    sys.path.append("..")
    main()
