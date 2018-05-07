#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np


class NodeBase(object):

    def __init__(self, name):
        self.name = name
        self.trainable = True
        self.values = None


class Variable(NodeBase):

    def __init__(self, name, shape):
        super(Variable, self).__init__(name)
        self.trainable = True
        self.values = np.array(shape)


class Constant(NodeBase):

    def __init__(self, name, value):
        super(Constant, self).__init__(name)
        self.trainable = False
        self.values = value


class PlaceHolder(NodeBase):

    def __init__(self, name, shape):
        super(PlaceHolder, self).__init__(name)
        self.trainable = False
        self.values = np.array(shape)
        self.next_list = []


class Edge(object):

    def __init__(self, src_node, dst_node, op):
        self.src_node = src_node
        self.dst_node = dst_node
        self.op = op


class Neurons(NodeBase):

    def __init__(self, name):
        super(Neurons, self).__init__(name)
        self.pre_list = []
        self.next_list = []


def add_flow(src, op, name):
    """
    在src节点后增加op形成新的神经元dst
    :param src: 源节点
    :param op: 操作
    ;param name: 新节点名字
    :return: dst
    """
    dst = Neurons(name=name)
    e = Edge(src_node=src, dst_node=dst, op=op)
    dst.pre_list.append(e)
    src.next_list.append(e)
    return dst
