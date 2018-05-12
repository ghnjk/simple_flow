#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
sys.path.append("..")
from simple_flow.flow import *
from simple_flow.nn import *
from simple_flow.model import *
from simple_flow.train import *
import numpy as np


def test_model():
    batch_size = 3
    x = PlaceHolder(name="x", shape=[batch_size, 1])
    y = PlaceHolder(name="y", shape=[batch_size, 1])
    a = Variable(name="a", shape=(1, ))
    b = Variable(name="b", shape=(1, ))
    ax = add_flow(x, Multify(a), name="a*x")
    ax_b = add_flow(ax, Add(b), name="a*x + b")
    ax_b_y = add_flow(ax_b, Sub(y), name="a*x + b - y")
    loss = add_flow(ax_b_y, Pow(n=2), name="(a*x + b - y)^2")
    loss = add_flow(loss, ReduceMean(), name="loss")
    model = Model(losses=loss, predicts=ax_b, optimizer=GradientDecent(learning_rate=0.001))
    feed_dict = {
        x: np.array([
            [10],
            [2],
            [1]
        ]),
        y: np.array([
            [40],
            [13],
            [10]
        ])
    }
    model.fit(feed_dict=feed_dict, max_train_itr=5000)
    print("predict: ", model.predict(feed_dict=feed_dict))
