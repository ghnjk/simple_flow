#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np


class NodeBase(object):

    def __init__(self, name):
        self.name = name
        self.trainable = True
        self.values = None
        self.next_edges = []
        self.pre_edges = []


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
