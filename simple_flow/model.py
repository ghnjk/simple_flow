#!/usr/bin/python
# -*- coding: UTF-8 -*-
import flow
import time


class Model(object):

    def __init__(self, losses, predicts, optimizer):
        if isinstance(losses, flow.Neurons):
            self.loss_node = losses
        else:
            self.loss_node = flow.Neurons(name="simple_flow/auto_gen/all_loss")
            for node in losses:
                flow.link_node(node, self.loss_node)
        self.predict_node = predicts
        self.predict_network = []
        if isinstance(self.predict_node, flow.Neurons):
            net = flow.NetWork()
            net.parse(self.predict_node)
            self.predict_network.append(net)
        else:
            for node in self.predict_node:
                net = flow.NetWork()
                net.parse(node)
                self.predict_network.append(net)
        self.optimizer = optimizer
        self.optimizer.mininize(self.loss_node)

    def fit(self, feed_dict, max_train_itr=10000, verbose=True):
        for i in range(max_train_itr):
            start = time.time()
            self.optimizer.prepare()
            self.optimizer.calc_gradient(feed_dict=feed_dict)
            apply_count = self.optimizer.apply_gradient()
            end = time.time()
            if verbose:
                use = round(end - start, 3)
                print("train iter %d/%d use %0.3lf sec:  loss: %s" % (
                    i + 1,
                    max_train_itr,
                    use,
                    str(self.loss_node.values)
                ))
            if apply_count <= 0:
                break

    def predict(self, feed_dict):
        if isinstance(self.predict_node, flow.Neurons):
            self.predict_network[0].forward_propagate(feed_dict=feed_dict)
            return self.predict_node.values
        else:
            res = []
            for net in self.predict_network:
                net.forward_propagate(feed_dict=feed_dict)
            for node in self.predict_node:
                res.append(node.values)
            return res
