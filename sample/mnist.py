#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_mnist_data():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets('MNIST_data', one_hot=True)


def build_model():
    from simple_flow.flow import PlaceHolder, add_flow, Constant
    from simple_flow.nn import Log, Softmax, Add, ReduceMean, Multify, Relu, ReduceSum
    from simple_flow.model import Model
    from simple_flow.train import GradientDecent
    from simple_flow.layer import dense
    x = PlaceHolder(name="img", shape=[None, 784])  # 28 * 28
    y = PlaceHolder(name="label", shape=[None, 10])  # 10=【0， 9】数字为x的概率label[n]=1 其他为0
    l_1 = dense(x, in_dims=784, out_dims=64, activation_fucntion=Relu(), name="dense_layer_1")
    predict = dense(l_1, in_dims=64, out_dims=10, activation_fucntion=Softmax(), name="predict_dense")
    # 计算交叉熵(Cross-Entropy) h = sigmae(-p(x) * log(predict(x)))
    epsilon = Constant(name="eplison_for_predict", value=[1e-8])
    log_predict = add_flow(
        add_flow(predict, Add(epsilon), name="predict + epsilon"),
        Log(), name="log(predict(x))")
    y_log_predict = add_flow(log_predict, Multify(y), name="y * log(predict)")
    loss = add_flow(y_log_predict, ReduceSum(axis=1), name="reduce_sum(y * log(p))")
    loss = add_flow(loss, Multify(Constant(name="-1", value=-1.0)), name="-reduce_sum(y * log(p))")
    loss = add_flow(loss, ReduceMean(), name="loss")
    model = Model(losses=loss, predicts=predict, optimizer=GradientDecent(lr_movement=0))
    return model, x, y


def calc_accuracy(model, feed_dict, label):
    predict = model.predict(feed_dict=feed_dict)
    predict = np.argmax(predict, axis=1)
    label = np.argmax(label, axis=1)
    succ_count = 0
    for i in range(len(predict)):
        if predict[i] == label[i]:
            succ_count += 1
    return succ_count / float(len(predict))


def main():
    # prepare data
    mnist = load_mnist_data()
    model, x, y = build_model()
    batch_size = 200
    epoch_count = 100
    loss_his = []
    accuracy_his = []
    test_feed_dict = {
        x: mnist.test.images
    }
    # train
    for epoch in range(epoch_count):
        x_data, y_data = mnist.train.next_batch(batch_size)
        model.fit(feed_dict={
            x: x_data,
            y: y_data
        }, max_train_itr=400, verbose=0)
        print("epoch ", epoch, "loss: ", model.loss_node.values)
        loss_his.append(model.loss_node.values)
        accuracy = int(calc_accuracy(model, test_feed_dict, mnist.test.labels) * 100)
        print("accuracy:", accuracy)
        accuracy_his.append(accuracy)
    # draw
    plt.subplot(211)
    plt.title("mnist loss")
    plt.ylabel("cross entropy loss")
    plt.plot(loss_his)
    plt.grid()
    plt.subplot(212)
    plt.title("mnist accuracy")
    plt.ylabel("accuracy(%)")
    plt.plot(accuracy_his)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    sys.path.append("..")
    main()
