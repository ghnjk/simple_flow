#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import nn


class NodeBase(object):

    def __init__(self, name):
        self.name = name
        self.trainable = True
        self.values = None
        self.pre_list = []
        self.next_list = []

    def __hash__(self):
        return id(self)


class Variable(NodeBase):

    def __init__(self, name, shape, initializer=nn.RandomaInitializer()):
        super(Variable, self).__init__(name)
        self.trainable = True
        self.values = initializer.get_vars(shape=shape)


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
    :param name: 新节点名字
    :return: dst
    """
    dst = Neurons(name=name)
    e = Edge(src_node=src, dst_node=dst, op=op)
    dst.pre_list.append(e)
    src.next_list.append(e)
    return dst


def link_node(src, dst):
    e = Edge(src_node=src, dst_node=dst, op=nn.Link())
    dst.pre_list.append(e)
    src.next_list.append(e)
    return dst


class NetWork(object):

    def __init__(self):
        self.in_nodes = []
        self.out_node = None
        self.flow_nodes = []
        self.node_id_sets = set()
        self.mpNode = {}
        self.mpGradient = {}
        self.mpNodeError = {}

    def parse(self, loss_node):
        """
        解析网络
        :param loss_node:  目标节点
        :return:
        """
        self.in_nodes = []
        self.flow_nodes = []
        self.mpNode = {}
        self.mpGradient = {}
        self.mpNodeError = {}
        self.node_id_sets = set()
        self.out_node = loss_node
        self._dfs_search_in_nodes(self.out_node)
        self._bfs_search_flows()

    def forward_propagate(self, feed_dict):
        """
        前向传播
        :param feed_dict: 用于替代place holder节点的数据
        :return:
        """
        for node in self.flow_nodes:
            if isinstance(node, Neurons):
                node.values = None
        for node in self.flow_nodes:
            if isinstance(node, PlaceHolder):
                if node not in feed_dict:
                    raise Exception("feed_dict lack variable for name: " + node.name)
                v = feed_dict[node]
                node.values = v
            else:
                v = node.values
            for e in node.next_list:
                dst = e.dst_node
                if id(dst) not in self.node_id_sets:
                    continue
                op = e.op
                if isinstance(op.get_trainable_w(), PlaceHolder):
                    # 支持边上为输入参数
                    w = op.get_trainable_w()
                    if w not in feed_dict:
                        raise Exception("feed_dict lack variable for name " + w.name)
                    w.values = feed_dict[w]
                    # print("w.values: ", w.values)
                nv = op.calc(v)
                if dst.values is None:
                    dst.values = nv
                else:
                    dst.values += nv
            # print("node: ", dst.name, " values:", dst.values)

    def backward_propagate(self):
        """
        反向传播
        :return:
        """
        self.mpNodeError = {}
        self.mpGradient = {}
        error = np.zeros(shape=self.out_node.values.shape) + 1.0
        self.mpNodeError[self.out_node] = error
        for i in range(len(self.flow_nodes) - 2, -1, -1):
            node = self.flow_nodes[i]
            error = None
            # 计算每条边的损失反向传播
            for e in node.next_list:
                dst = e.dst_node
                if dst not in self.mpNodeError:
                    # 下游节点不可训练， 则本节点也不能训练
                    error = None
                    break
                dst_error = self.mpNodeError[dst]
                if e.op.get_trainable_w() is not None and e.op.get_trainable_w().trainable:
                    op_gradient = e.op.calc_gradient(x=node.values, y=dst.values, y_errors=dst_error)
                    # print("mpGradient: ", e.op.get_trainable_w().name, ", gradient: ", op_gradient)
                    self.mpGradient[e.op] = op_gradient
                bp_error = e.op.backpropagete_error(x=node.values, y=dst.values, y_errors=dst_error)
                if error is None:
                    error = bp_error
                else:
                    error += bp_error
            # print("node: ", node.name, "error: ", error)
            if error is not None and node.trainable:
                self.mpNodeError[node] = error

    def apply_gradient(self, lr):
        """
        根据计算出来的梯度， 对所有的变量做 -lr*gradient
        :param lr:
        :return:
        """
        for node in self.flow_nodes:
            for e in node.next_list:
                w = e.op.get_trainable_w()
                if e.op in self.mpGradient and w is not None and w.trainable:
                    # print("w(", w.name, "): ", w.values, "gradients: ", self.mpGradient[e.op])
                    w.values -= lr * self.mpGradient[e.op]

    def _dfs_search_in_nodes(self, node):
        node_id = id(node)
        if node_id in self.mpNode.keys():
            return
        self.mpNode[node_id] = {
            "node": node,
            "in_d": len(node.pre_list)
        }
        self.node_id_sets.add(node_id)
        if self.mpNode[node_id]["in_d"] == 0:
            self.in_nodes.append(node)
        for e in node.pre_list:
            self._dfs_search_in_nodes(e.src_node)
        if len(node.pre_list) > 0 and not isinstance(node, Neurons):
            raise Exception("inner node %s must be instance of Neurons" % node.anme)

    def _bfs_search_flows(self):
        q = []
        for node in self.in_nodes:
            node_id = id(node)
            q.append(node)
            self.flow_nodes.append(node)
            del self.mpNode[node_id]
        while len(q) > 0:
            node = q.pop(0)
            for e in node.next_list:
                c = e.dst_node
                c_id = id(c)
                if c_id not in self.node_id_sets:
                    continue
                self.mpNode[c_id]["in_d"] -= 1
                if self.mpNode[c_id]["in_d"] == 0:
                    q.append(c)
                    self.flow_nodes.append(c)
                    del self.mpNode[c_id]
        if len(self.mpNode.keys()) != 0:
            raise Exception("invalid graph.may circle nodes in the network.")

    def __str__(self):
        s = "{network structure:"
        s += "in_nodes: "
        for node in self.in_nodes:
            s += node.name
            s += ", "
        s += "\n"
        s += "out_node: " + self.out_node.name
        s += "\nflow nodes: "
        for node in self.flow_nodes:
            s += node.name
            s += ", "
        s += "}"
        return s
