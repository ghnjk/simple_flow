#!/usr/bin/python
# -*- coding: UTF-8 -*-
import flow


class Optimizer(object):

    def __init__(self, name, learning_rate):
        self.name = name
        self.learning_rate = learning_rate

    def mininize(self, loss):
        """
        最小化loss节点
        :param loss: loss node
        :return: None
        """
        pass

    def prepare(self):
        """
        每次训练前准备
        :return:
        """
        pass

    def calc_gradient(self, feed_dict):
        """
        计算各个参数的梯度
        :param feed_dict: 输入参数
        :return: 是否需要更新
        """
        pass

    def apply_gradient(self):
        """
        将计算出来的梯度更新到所有参数中
        :return: None
        """
        pass


class GradientDecent(Optimizer):

    def __init__(self, name=None, learning_rate=0.001, lr_movement=1e-8):
        super(GradientDecent, self).__init__(name=name, learning_rate=learning_rate)
        self.loss_node = None
        self.lr_movement = lr_movement
        self.net = flow.NetWork()

    def mininize(self, loss):
        """
        最小化loss节点
        :param loss: loss node
        :return: None
        """
        self.loss_node = loss
        self.net.parse(self.loss_node)

    def prepare(self):
        """
        每次训练前准备
        :return:
        """
        self.learning_rate -= self.lr_movement
        if self.learning_rate < 1e-6:
            self.learning_rate = 1e-6

    def calc_gradient(self, feed_dict):
        """
        计算各个参数的梯度
        :param feed_dict: 输入参数列表
        :return: 是否需要更新
        """
        # 前向传播
        self.net.forward_propagate(feed_dict)
        self.net.backward_propagate()

    def apply_gradient(self):
        """
        将计算出来的梯度更新到所有参数中
        :return: None
        """
        self.net.apply_gradient(self.learning_rate)
