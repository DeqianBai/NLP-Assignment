#！/usr/bin/env python
#  -*- coding:utf-8 -*-
#  author:dabai time:2019/9/13

import numpy as np
import gzip
import os
from urllib.request import urlretrieve
from scipy import signal

def get_data(number_of_classes=2, seed=42, number_of_features=5, number_of_examples=1000,train_set_ratio=0.7):
    np.random.seed(seed)

    # 对每一类别生成样本
    data = []
    for i in range(number_of_classes):
        h = np.mat(np.random.random((number_of_features, number_of_features))) * 0.2
        features = np.random.multivariate_normal(mean=np.random.random(number_of_features),
                                                 cov = h.T * h + 0.03 * np.mat(np.eye(number_of_classes)), # 随机生成一个对称矩阵作为协方差矩阵，有可能不正定
                                                 check_valid="raise", # # 如果不正定，抛出异常
                                                 size=number_of_examples) # 样本数

        labels = np.array([[int(i==j) for j in range(number_of_classes)]] * number_of_examples)
        data.append(np.c_[features,labels])

    # 把各个类别的样本合在一起
    data = np.concatenate(data, axis=0)

    # 随机打乱样本顺序
    np.random.shuffle(data)

    # 计算训练集样本数量
    train_set_size = int(number_of_examples * train_set_ratio)

    # 将训练集和测试集、特征和标签分开
    return (data[:train_set_size, :-number_of_classes],
            data[:train_set_size,-number_of_classes :],
            data[train_set_size:, :-number_of_classes],
            data[train_set_size:,-number_of_classes:])


def get_sequence_data(number_of_classes=2, dimension=10, length = 10, number_of_examples=1000,train_set_ratio=0.7,seed=42):
    """
    生成两类序列数据
    :param number_of_classes:
    :param dimension:
    :param length:
    :param number_of_examples:
    :param train_set_ration:
    :param seed:
    :return:
    """
    xx = []
    xx.append(np.sin(np.arange(0,10, 10/length)))
    xx.append(np.array(signal.square(np.arange(0,10,10/length))))

    data = []
    for i in range(number_of_classes):
        x = xx[i]
        for j in range(number_of_examples):
            sequence = x + np.random.normal(0,0.3, (dimension, len(x)))
            label = np.array([int(i==j) for j in range(number_of_classes)])

            data.append(np.c_[sequence.reshape(1,-1), label.reshape(1,-1)])

        # 把各个类别的样本合在一起
        data = np.concatenate(data, axis=0)

        # 随机打乱样本顺序
        np.random.shuffle(data)

        # 计算训练集样本数量
        train_set_size = int(number_of_examples * train_set_ratio)

        # 将训练集和测试集、特征和标签分开
        return (data[:train_set_size, :-number_of_classes],
                data[:train_set_size, -number_of_classes:],
                data[train_set_size:, :-number_of_classes],
                data[train_set_size:, -number_of_classes:])


def construct_pow2(x):
    """
    利用特征构造二次交互项特征
    :param x:
    :return:
    """
    m = x * x.T
    x_2 = []
    for i in range(len(x)):
        for j in range(i):
            x_2.append(m[i, j])

    return np.mat(x_2).T


def fill_diagonal(to_be_filled, filler):
    """
    将filler矩阵填充在to_be_filled的对角线上
    :param to_be_filled:
    :param filler:
    :return:
    """
    assert  to_be_filled.shape[0] / filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r,c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


def mnist(path=None):

    # :param path (str): Directory containing MNIST. Default is
    #         /home/USER/data/mnist or C:\Users\USER\data\mnist.
    #         Create if non-existant. Download any missing files.
    # :return:  Tuple of (train_images, train_labels, test_images, test_labels), each a matrix.
    #           Rows are examples. Columns of images are pixel values.
    #           Columns of labels are a one-hot encoding of the correct class.
    # :param path:
    # :return:


    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'),'data','mnist')

    # creat path if it does not exist
    os.makedirs(path,exist_ok=True)

    # Download any missing files

    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path,file))
            print("Downland %s to %s" % (file,path))

    def _images(path):
        """
        Return images loaded locally.
        :param path:
        :return:
        """
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        """

        :param path:
        :return:
        """
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(),'B',offset=8)

        def _onehot(integer_labels):
            """Return matrix whose rows are one-hot encodings of integers"""
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols),dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path,files[1]))
    test_images = _images(os.path.join(path,files[2]))
    test_labels = _labels(os.path.join(path,files[3]))

    return train_images,train_labels,test_images,test_labels