#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pytest
import sys
sys.path.append("..")
from simple_flow.flow import *
from simple_flow.nn import *


def test_network_parse():
    x = PlaceHolder(name="x", shape=[None, 10])
    y = add_flow(x,
                 Multify(Variable(name="w1", shape=[10, 2])),
                 "multify_1")
    y1 = add_flow(y,
                  Add(Variable(name="bias1", shape=[None, ])),
                  "plus_1")
    y = add_flow(x,
                 Multify(Variable(name="w2", shape=[10, 2])),
                 "multify_2")
    y2 = add_flow(y,
                  Add(Variable(name="bias2", shape=[None, ])),
                  "plus_2")
    link_node(y2, y1)
    y = Neurons(name="loss")
    link_node(y1, y)
    link_node(y2, y)
    net = NetWork()
    net.parse(y)
    print(net)


def test_circle_network():
    x = PlaceHolder(name="x", shape=[None, 10])
    y = add_flow(x,
                 Multify(Variable(name="w1", shape=[10, 2])),
                 "multify_1")
    y1 = add_flow(y,
                  Add(Variable(name="bias1", shape=[None, ])),
                  "plus_1")
    y = add_flow(x,
                 Multify(Variable(name="w2", shape=[10, 2])),
                 "multify_2")
    y2 = add_flow(y,
                  Add(Variable(name="bias2", shape=[None, ])),
                  "plus_2")
    link_node(y2, y1)
    link_node(y1, y2)
    y = Neurons(name="loss")
    link_node(y1, y)
    link_node(y2, y)
    net = NetWork()
    pytest.raises(Exception, net.parse, y)
