#!/usr/bin/python
# -*- coding: UTF-8 -*-
import flow
import nn


def dense(in_node, in_dims, out_dims, activation_fucntion=None, name=None):
    """
    添加全连接网络
    :param in_node:
    :param in_dims:
    :param out_dims:
    :param activation_fucntion:
    :param name:
    :return:
    """
    if name is None:
        name = "dense_" + str(id(in_node))
    w = flow.Variable(name="%s_w" % name,
                      shape=[in_dims, out_dims])
    bias = flow.Variable(name="%s_bias" % name,
                         shape=[out_dims, ],
                         initializer=nn.ConstantInitializer(0.1))
    ax = flow.add_flow(in_node, nn.MatMul(w), name="%s.dot(w)" % name)
    ax_b = flow.add_flow(ax, nn.Add(bias), name="%s.dot(w) + bias" % name)
    if activation_fucntion is None:
        return ax_b
    else:
        return flow.add_flow(ax_b, activation_fucntion, name="act_func(%s)" % name)
