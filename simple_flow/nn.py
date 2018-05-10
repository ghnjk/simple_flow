#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np


class OperatorBase(object):

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        pass

    def derivative(self, x, y):
        """
        导数
        :param y:
        :param x:
        :return:
        """
        pass

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None


class Multify(OperatorBase):

    def __init__(self, w):
        self.w = w

    def calc(self, x):
        """
        计算
        :param x:
        :return inputes * w
        """
        return x * self.w.values

    def derivative(self, x, y):
        """
        导数
        :param y:
        :param x:
        :return:
        """
        return x

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return self.w

    def __str__(self):
        return "x * w" + str(np.array(self.w.values).shape)


class MatMul(OperatorBase):

    def __init__(self, w):
        self.w = w

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        return np.dot(x, self.w.values)

    def derivative(self, x, y):
        """
        导数
        :param y:
        :param x:
        :return:
        """
        return x

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return self.w

    def __str__(self):
        return "matmul(x, w" + str(np.array(self.w.values).shape) + ")"


class Add(OperatorBase):

    def __init__(self, w):
        self.w = w

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        return x + self.w.values

    def derivative(self, x, y):
        """
        导数
        :param y:
        :param x:
        :return:
        """
        return 1

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return self.w

    def __str__(self):
        return "x + w" + str(np.array(self.w.values).shape)


class Link(OperatorBase):

    def __init__(self):
        self.w = None

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        return x

    def derivative(self, x, y):
        """
        导数
        :param y:
        :param x:
        :return:
        """
        return 1

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "link x"


class ReduceSum(OperatorBase):

    def __init__(self, axis=None):
        self.w = None
        self.axis = axis

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        if self.axis is None:
            return np.sum(x)
        else:
            return np.sum(x, axis=self.axis)

    def derivative(self, x, y):
        """
        导数
        :param y:
        :param x:
        :return:
        """
        return 1

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "link x"


class Pow(OperatorBase):

    def __init__(self, n):
        self.n = n

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        return np.power(x, self.n)

    def derivative(self, x, y):
        """
        导数
        :param y:
        :param x:
        :return:
        """
        return self.n * np.power(x, self.n - 1)

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "x ^ " + str(self.n)


class Softmax(OperatorBase):

    def __init__(self):
        pass

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def derivative(self, x, y):
        """
        导数
        :param y:
        :param x:
        :return:
        """
        m = y.reshape(-1, 1)
        return np.diag(y) - np.dot(m, m.T)

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "softmax(x)"


class Sigmoid(OperatorBase):

    def __init__(self):
        pass

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        return 1.0/(1.0+np.exp(-x))

    def derivative(self, x, y):
        """
        导数
        :param x:
        :param y:
        :return:
        """
        return y * (1 - y)

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "sigmoid(x)"


class VarInitializer(object):

    def get_vars(self, shape, dtype=np.float32):
        return np.array([], dtype=dtype)


class RandomaInitializer(VarInitializer):

    def __init__(self, scale=0.3):
        self.scale = scale

    def get_vars(self, shape, dtype=np.float32):
        return np.random.normal(0, self.scale, size=shape)


class ConstantInitializer(VarInitializer):

    def __init__(self, v=0.0):
        self.v = v

    def get_vars(self, shape, dtype=np.float32):
        return np.zeros(shape=shape, dtype=dtype) + self.v
