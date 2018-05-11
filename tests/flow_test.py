#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pytest
import sys
sys.path.append("..")
from simple_flow.flow import *
from simple_flow.nn import *


def test_network_parse():
    batch_size = 2
    x = PlaceHolder(name="x", shape=[batch_size, 10])
    y = add_flow(x,
                 Multify(Variable(name="w1", shape=[10, 2])),
                 "multify_1")
    y1 = add_flow(y,
                  Add(Variable(name="bias1", shape=[batch_size, ])),
                  "plus_1")
    y = add_flow(x,
                 Multify(Variable(name="w2", shape=[10, 2])),
                 "multify_2")
    y2 = add_flow(y,
                  Add(Variable(name="bias2", shape=[batch_size, ])),
                  "plus_2")
    link_node(y2, y1)
    y = Neurons(name="loss")
    link_node(y1, y)
    link_node(y2, y)
    net = NetWork()
    net.parse(y)
    print(net)


def test_circle_network():
    batch_size = 2
    x = PlaceHolder(name="x", shape=[batch_size, 10])
    y = add_flow(x,
                 Multify(Variable(name="w1", shape=[10, 2])),
                 "multify_1")
    y1 = add_flow(y,
                  Add(Variable(name="bias1", shape=[batch_size, ])),
                  "plus_1")
    y = add_flow(x,
                 Multify(Variable(name="w2", shape=[10, 2])),
                 "multify_2")
    y2 = add_flow(y,
                  Add(Variable(name="bias2", shape=[batch_size, ])),
                  "plus_2")
    link_node(y2, y1)
    link_node(y1, y2)
    y = Neurons(name="loss")
    link_node(y1, y)
    link_node(y2, y)
    net = NetWork()
    pytest.raises(Exception, net.parse, y)


def test_network_forward_propagate():
    batch_size = 2
    x = PlaceHolder(name="x", shape=[batch_size, 10])
    y = add_flow(x,
                 MatMul(Variable(name="w1", shape=[10, 2])),
                 "multify_1")
    y1 = add_flow(y,
                  Add(Variable(name="bias1", shape=[batch_size, ])),
                  "plus_1")
    y = add_flow(x,
                 MatMul(Variable(name="w2", shape=[10, 2])),
                 "multify_2")
    y2 = add_flow(y,
                  Add(Variable(name="bias2", shape=[batch_size, ])),
                  "plus_2")
    link_node(y2, y1)
    y = Neurons(name="loss")
    link_node(y1, y)
    link_node(y2, y)
    net = NetWork()
    net.parse(y)
    feed_dict = {
        x: [[2] * 10] * batch_size
    }
    net.forward_propagate(feed_dict=feed_dict)
    print("loss: ", y.values)


def test_network_forward_propagate_2():
    batch_size = 2
    x = PlaceHolder(name="x", shape=[batch_size, 2])
    y = add_flow(x,
                 MatMul(
                     Variable(name="w1", shape=[2, 2],
                              initializer=ConstantInitializer(v=np.array([
                                  [2, 3],
                                  [3, 2]
                              ]))
                              )
                 ),
                 "multify_1")
    y = add_flow(y,
                 Add(
                     Variable(name="bias1", shape=[batch_size, 2],
                              initializer=ConstantInitializer(v=np.array([
                                  [3, 4],
                                  [1, 2]
                              ])))
                 ),
                 "plus_1")
    y = add_flow(y,
                 ReduceSum(),
                 "reduce_sum")
    net = NetWork()
    net.parse(y)
    feed_dict = {
        x: np.array([
            [3, 0],
            [0, 3]
        ])
    }
    net.forward_propagate(feed_dict=feed_dict)
    assert 40 == y.values
    print("loss: ", y.values)


def test_backward_propagate():
    batch_size = 2
    x = PlaceHolder(name="x", shape=[batch_size, 2])
    y = add_flow(x,
                 MatMul(
                     Variable(name="w1", shape=[2, 2],
                              initializer=ConstantInitializer(v=np.array([
                                  [2, 3],
                                  [3, 2]
                              ]))
                              )
                 ),
                 "multify_1")
    y = add_flow(y,
                 Add(
                     Variable(name="bias1", shape=[batch_size, 2],
                              initializer=ConstantInitializer(v=np.array([
                                  [3, 4],
                                  [1, 2]
                              ])))
                 ),
                 "plus_1")
    y = add_flow(y,
                 ReduceSum(),
                 "reduce_sum")
    net = NetWork()
    net.parse(y)
    feed_dict = {
        x: np.array([
            [3, 0],
            [0, 3]
        ])
    }
    print("net: ", str(net))
    net.forward_propagate(feed_dict=feed_dict)
    assert 40 == y.values
    net.backward_propagate()
    for node in net.flow_nodes:
        for e in node.next_list:
            if e.op in net.mpGradient and net.mpGradient[e.op] is not None:
                print("gradient for ", e.op.w.name, " is ", net.mpGradient[e.op])
