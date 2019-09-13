#！/usr/bin/env python
#  -*- coding:utf-8 -*-
#  author:dabai time:2019/9/11

import numpy as np
from miniflow import *


# 下载数据
data = load_boston()

# 数据与标签
X_ = data['data']
y_ = data['target']
# Normalization
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]

# 隐藏层结点
n_hidden = 10
n_hidden_2 = 10

W1_, b1_ = np.random.randn(n_features, n_hidden), np.zeros(n_hidden)
W2_, b2_ = np.random.randn(n_hidden, 1), np.zeros(1)
# W3, b3 = np.random.randn(n_hidden_2, 1), np.zeros(1)

# Build a graph connection

## 1st. Build Nodes in this graph
X, y = Input(name='X'), Input(name='y')     # tensorflow -> placeholder
W1, b1 = Input(name='W1'), Input(name='b1')
W2, b2 = Input(name='W2'), Input(name='b2')
# W3, b3 = Input(name='W3'), Input(name='b3')

# 2nd build connection relationship
# MSE(Linear(Sigmoid(Linear(X, W1, b1)), W2, b2), y)
# 目标是根据Input调整weight和biases（W1, b1, W2, b2），使得loss最小。

L1 = Linear(X, W1, b1)
S1 = Sigmoid(L1)
yhat = Linear(S1, W2, b2)
loss = MSE(yhat, y)

# -> feed_dict
input_node_with_value = {
        X: X_,
        y: y_,
        W1: W1_,
        W2: W2_,
        b1: b1_,
        b2: b2_
    }

graph = topological_sort(input_node_with_value)

def main():

    losses = []
    epochs = 5000

    batch_size = 64

    steps_per_epoch = X_.shape[0] // batch_size

    for i in range(epochs):
        loss = 0
        for batch in range(steps_per_epoch):
            # 不需要用sklearn learning时:
            # indices = np.random.choice(range(X_.shape[0]),size = 10,replace=True)
            # X_batch = X_[indices]
            # y_batch = y_[indices]
            # 需要用sklearn learning 时:
            # X_batch, y_batch = resample(X_,y_,n_samples=batch_size)
            #
            # X.value = X_batch
            # y.value = y_batch

            forward_and_backward(graph)

            learning_rate = 1e-3

            sgd_update(trainable_nodes=[W1, W2, b1, b2], learning_rate  = learning_rate)
            loss += graph[-1].value

        if i % 100 ==0:
            print('Epoch: {}， loss = {: .3f}'.format(i+1,loss/steps_per_epoch))
            losses.append(loss)

    plt.plot(losses)
    plt.show()



if __name__ == '__main__':
    main()