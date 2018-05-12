#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
sys.path.append("..")
from simple_flow.flow import *
from simple_flow.nn import *
from simple_flow.model import *
from simple_flow.train import *


def test_power():
    batch_size = 3
    x = PlaceHolder(name="x", shape=[batch_size, 1])
    b = Variable(name="b", shape=(1, ), initializer=ConstantInitializer(np.array([100.0])))
    x_b = add_flow(x, Add(b), name="x + b")
    loss = add_flow(x_b, Pow(n=2), name="(x + b)^2")
    loss = add_flow(loss, ReduceMean(), name="loss")
    model = Model(losses=loss, predicts=loss, optimizer=GradientDecent(learning_rate=0.001))
    feed_dict = {
        x: np.array([
            [10],
            [2],
            [1]
        ])
    }
    model.fit(feed_dict=feed_dict, max_train_itr=10000)
    print("predict: ", model.predict(feed_dict=feed_dict))
    print("b: ", b.values)