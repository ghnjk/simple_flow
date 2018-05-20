# -*- coding: utf-8 -*-
# @File: gradient_decent.py
# @Author: jkguo
# @Create: 2018-05-20
import matplotlib.pyplot as plt
import numpy as np


def show_decent_proces():
    x0 = -1.5
    lr = 0.1
    x = np.linspace(-2, 2, 1000)
    y = np.square(x)
    plt.ion()
    for epoch in range(20):
        dx = 2 * x0
        a = dx
        y0 = np.square(x0)
        b = y0 - a * x0
        lx = np.linspace(x0 - 0.5, x0 + 0.5, 200)
        ly = a * lx + b
        plt.cla()
        plt.plot(x, y)
        plt.plot(lx, ly)
        plt.axhline()
        plt.axvline()
        plt.draw()
        plt.pause(0.1)
        x0 -= lr * dx
        print(x0)


if __name__ == '__main__':
    show_decent_proces()
