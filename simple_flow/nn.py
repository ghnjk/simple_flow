#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np


def calc_ele_count(shape):
    res = 1
    for i in shape:
        res *= i
    return res


class OperatorBase(object):

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        pass

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        pass

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        pass

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __hash__(self):
        return id(self)


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

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        if y_errors.shape == self.w.values.shape:
            return x * y_errors
        elif self.w.values.shape == (1, ):
            # print("np.sum", np.sum(y_errors))
            return np.sum(x * y_errors)
        else:
            raise Exception("calc_gradient failed")

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        return self.w.values * y_errors

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

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return np.dot(x.T,  y_errors)

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        return np.dot(y_errors, self.w.values.T)

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

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return np.sum(y_errors, axis=0)

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        if x.shape == y_errors.shape:
            return y_errors
        elif x.shape == (1, ):
            return np.sum(y_errors)
        else:
            raise Exception("calc_gradient failed")

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return self.w

    def __str__(self):
        return "x + w" + str(np.array(self.w.values).shape)


class Sub(OperatorBase):

    def __init__(self, w):
        self.w = w

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        # print("x: ", x, "w: ", self.w.values)
        return x - self.w.values

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return - np.sum(y_errors, axis=0)

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        if x.shape == self.w.values.shape:
            return y_errors
        elif x.shape == (1, ):
            return np.sum(y_errors)
        else:
            raise Exception("calc_gradient failed")

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return self.w

    def __str__(self):
        return "x - w" + str(np.array(self.w.values).shape)


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

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return None

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        return y_errors

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

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return None

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        if self.axis is None:
            return np.zeros(shape=x.shape, dtype=np.float32) + y_errors
        else:
            ns = [x.shape[self.axis]]
            for i in y_errors.shape:
                ns.append(i)
            a = np.empty(ns)
            a[:] = y_errors
            for i in range(self.axis):
                a = np.swapaxes(a, i, i + 1)
            return a

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "sum x"


class ReduceMean(OperatorBase):

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
            return np.mean(x)
        else:
            return np.mean(x, axis=self.axis)

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return None

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        if self.axis is None:
            return np.zeros(shape=x.shape, dtype=np.float32) + y_errors / float(calc_ele_count(x.shape))
        else:
            ns = [x.shape[self.axis]]
            for i in y_errors.shape:
                ns.append(i)
            a = np.empty(ns)
            a[:] = y_errors / float(x.shape[self.axis])
            for i in range(self.axis):
                a = np.swapaxes(a, i, i + 1)
            return a

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "mean x"


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

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return None

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        # print("x: ", x)
        # print("y_errors: ", y_errors, "pre_errors: ", self.n * np.power(x, self.n - 1) * y_errors)
        return self.n * np.power(x, self.n - 1) * y_errors

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "x ^ " + str(self.n)


class Log(OperatorBase):

    def __init__(self):
        pass

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        return np.log(x)

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return None

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        return y_errors / x

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "log(x)"


class Relu(OperatorBase):

    def __init__(self):
        pass

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        # print("x: ", x[0])
        # print("relu: ", np.maximum(x[0], 0))
        return np.maximum(x, 0)

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return None

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        error = np.array(y_errors)
        error[x < 0] = 0
        # print(x[0])
        # print(error[0])
        return error

    def get_trainable_w(self):
        """
        获取可被训练的参数
        :return:
        """
        return None

    def __str__(self):
        return "relu(x)"


class Softmax(OperatorBase):

    def __init__(self):
        pass

    def calc(self, x):
        """
        计算
        :param x:
        :return:
        """
        tmp = x - np.max(x, axis=1).reshape(-1, 1)
        e_x = np.exp(tmp)
        return e_x / e_x.sum(axis=1).reshape(-1, 1)

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return None

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        # m = y.reshape(-1, 1)
        # return (np.diag(y) - np.dot(m, m.T)) * y_errors
        delta = y * y_errors
        sm = delta.sum(axis=1, keepdims=True)
        delta -= y * sm
        return delta

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

    def calc_gradient(self, x, y, y_errors):
        """
        计算参数梯度 dloss/dw
        :param y:
        :param x:
        :param y_errors:
        :return:
        """
        return None

    def backpropagete_error(self, x, y, y_errors):
        """
        反向传播， 计算x的损失量
        :param x:
        :param y:
        :param y_errors:
        :return:
        """
        return y * (1 - y) * y_errors

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

    def __init__(self, v):
        self.v = np.array(v)

    def get_vars(self, shape, dtype=np.float32):
        return np.zeros(shape=shape, dtype=dtype) + self.v
