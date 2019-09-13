#！/usr/bin/env python
#  -*- coding:utf-8 -*-
#  author:dabai time:2019/9/13

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

from node import *
from optimizer import *

mnist = input_data.read_data_sets("F:/ML/Deep Learning/NLP/Course/2019-summer/Lesson10/train_pic/mnist_dataset/",one_hot=True)
train_x = mnist.train.images
train_y = mnist.train.labels

test_x = mnist.test.images
test_y = mnist.test.labels


# 构造多分类逻辑回归计算图，输入变量
# 构造多层全连接神经网络的计算图
X = Variable((784,1), trainable=False) # 10维特征向量

# 第一隐藏层12个神经元
hidden_1 = ReLU(
    Add(
        MatMul(
            Variable((90,784),True), # 90x784权值矩阵
            X
        ),
        Variable((90,1),True) # 90维偏置向量
    )
)

# 第二隐藏层8个神经元
hidden_2 = ReLU(
    Add(
        MatMul(
            Variable((20,90),True), # 20x90权值矩阵
            hidden_1
        ),
        Variable((20,1), True) # 20维偏置向量
    )
)


# 输出层6个神经元
logits = Add(
    MatMul(
        Variable((10,20),True), # 10x20权值矩阵
        hidden_2
    ),
    Variable((10,1),True) # 6维偏置向量
)

prob = SoftMax(logits)

# 训练标签
label = Variable((10,1), trainable=False)

# 交叉熵损失
loss = CrossEntropyWithSoftMax(logits,label)

# 绘制计算图
default_graph.draw()

# Adam优化器
optimizer = Adam(default_graph,loss, 0.01 ,batch_size=32)

# 训练


def main():
    global truth, pred
    for e in range(6):

        # 每个epoch 在测试上评估模型正确率
        probs = []
        losses = []
        for i in range(len(test_x)):
            X.set_value(np.mat(test_x[i,:]).T)
            label.set_value(np.mat(test_y[i,:]).T)

            # 前向传播计算概率
            prob.forward()
            probs.append(prob.value.A1)

            # 计算损失值
            loss.forward()
            losses.append(loss.value[0, 0])

        # 取概率最大的类别为预测类别
        pred = np.argmax(np.array(probs), axis=1)
        truth = np.argmax(test_y, axis=1)
        accuracy = accuracy_score(truth, pred)

        print("Epoch:{:d}, 测试集损失值： {:.3f}, 测试集正确率：{:.2f}%".format(e + 1, np.mean(losses), accuracy * 100))

        for i in range(len(train_x)):
            X.set_value(np.mat(train_x[i, :]).T)
            label.set_value(np.mat(train_y[i, :]).T)

            # 优化
            optimizer.one_step()

            # 计算Mini Batch上的损失
            if i % 500 == 0:
                loss.forward()
                print("Iteration: {:d}, Mini Batch 损失值：{:.3f}".format(i + 1, loss.value[0, 0]))

    print("验证集正确率：{:.3f}".format(accuracy_score(truth,pred)))
    print(classification_report(truth,pred))


if __name__ == '__main__':
    main()