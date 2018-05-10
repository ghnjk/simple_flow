#!/usr/bin/python
# -*- coding: UTF-8 -*-
import flow


class Optimizer(object):

    def __init__(self, name):
        self.name = name

    def mininize(self, loss, learning_rate):
        """
        最小化loss节点
        :param loss: loss node
        :param learning_rate: lr
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

    def __init__(self, name = None):
        super(GradientDecent, self).__init__(name=name)
        self.lr = 0.001
        self.loss_node = None
        self.lr_movement = 1e-6
        self.net = flow.NetWork()

    def mininize(self, loss, learning_rate=0.001):
        """
        最小化loss节点
        :param loss: loss node
        :param learning_rate: lr
        :return: None
        """
        self.loss_node = loss
        self.lr = learning_rate
        self.net.parse(self.loss_node)

    def prepare(self):
        """
        每次训练前准备
        :return:
        """
        self.lr -= self.lr_movement

    def calc_gradient(self, feed_dict):
        """
        计算各个参数的梯度
        :param feed_dict: 输入参数列表
        :return: 是否需要更新
        """
        # 前向传播
        self.net.forward_propagate(feed_dict)


    def apply_gradient(self):
        """
        将计算出来的梯度更新到所有参数中
        :return: None
        """
        pass
