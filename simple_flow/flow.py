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
    :param name: 新节点名字
    :return: dst
    """
    dst = Neurons(name=name)
    e = Edge(src_node=src, dst_node=dst, op=op)
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

    def parse(self, loss_node):
        self.in_nodes = []
        self.flow_nodes = []
        self.mpNode = {}
        self.node_id_sets = set()
        self.out_node = loss_node
        self._dfs_search_in_nodes(self.out_node)
        self._bfs_search_flows()

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
        for p in node.pre_list:
            self._dfs_search_in_nodes(p)

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
