#！/usr/bin/env python
#  -*- coding:utf-8 -*-
#  author:dabai time:2019/9/13

from graph import Graph, default_graph
from util import *


class Node:
    """
    计算图节点类基类
    """

    def __init__(self, *parents):
        self.parents = parents  # 父节点列表
        self.children = []  # 子节点列表
        self.value = None  # 本节点的值
        self.jacobi = None  # 结果节点对本节点的雅可比矩阵
        self.graph = default_graph  # 计算图对象，默认为全局对象default_graph

        # 将本节点添加到父节点的子节点列表中
        for parent in self.parents:
            parent.children.append(self)

        # 将本节点添加到计算图中
        self.graph.add_node(self)

    def set_graph(self, graph):
        """
        设置计算图
        """
        assert isinstance(graph, Graph)
        self.graph = graph

    def get_parents(self):
        """
        获取本节点的父节点
        """
        return self.parents

    def get_children(self):
        """
        获取本节点的子节点
        """
        return self.children

    def forward(self):
        """
        前向传播计算本节点的值，若父节点的值未被计算，则递归调用父节点的forward方法
        """
        for node in self.parents:
            if node.value is None:
                node.forward()

        self.compute()

    def compute(self):
        """
        抽象方法，根据父节点的值计算本节点的值
        """
        pass

    def get_jacobi(self, parent):
        """
        抽象方法，计算本节点对某个父节点的雅可比矩阵
        """
        pass

    def backward(self, result):
        """
        反向传播，计算结果节点对本节点的雅可比矩阵
        """
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                self.jacobi = np.mat(np.zeros((result.dimension(), self.dimension())))

                for child in self.get_children():
                    if child.value is not None:

                        try:
                            self.jacobi += child.backward(result) * child.get_jacobi(self)
                        except ValueError as e:
                            print("雅可比矩阵形状不容：{:s}".format(str(e)))

        return self.jacobi

    def clear_jacobi(self):
        """
        清空结果节点对本节点的雅可比矩阵
        """
        self.jacobi = None

    def dimension(self):
        """
        返回本节点的值展平成向量后的维数。展平方式固定式按行排列成一列。
        """
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        """
        返回本节点的值作为矩阵的形状：（行数，列数）
        """
        return self.value.shape

    def reset_value(self, recursive=True):
        """
        重置本节点的值，并递归重置本节点的下游节点的值
        """

        self.value = None

        if recursive:
            for child in self.children:
                child.reset_value()


class Variable(Node):
    """
    变量节点
    变量节点
    """

    def __init__(self, dim, init=False, trainable=True):
        """
        变量节点没有父节点，构造函数接受变量的维数，以及变量是否参与训练的标识
        """
        Node.__init__(self)
        self.dim = dim

        # 如果需要初始化，则以正态分布随机初始化变量的值
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))

        # 变量节点是否参与训练
        self.trainable = trainable

    def set_value(self, value):
        """
        为变量赋值
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim

        # 本节点的值被改变，重置所有下游节点的值
        self.reset_value()
        self.value = value


class Add(Node):
    """
    （多个）矩阵加法
    """

    def compute(self):
        # assert len(self.parents) == 2 and self.parents[0].shape() == self.parents[1].shape()
        self.value = np.mat(np.zeros(self.parents[0].shape()))

        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):
        return np.mat(np.eye(self.dimension()))  # 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵


class MatMul(Node):
    """
    矩阵乘法
    """

    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        """
        将矩阵乘法视作映射，求映射对参与计算的矩阵的雅克比矩阵。
        """

        # 很神秘，靠注释说不明白了
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Dot(Node):
    """
    向量内积
    """

    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].dimension() == self.parents[1].dimension()
        self.value = self.parents[0].value.T * self.parents[1].value  # 1x1矩阵（标量），为两个父节点的内积

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return self.parents[1].value.T
        else:
            return self.parents[0].value.T


class Logistic(Node):
    """
    对矩阵的元素施加Logistic函数
    """

    def compute(self):
        x = self.parents[0].value
        self.value = np.mat(1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))  # 对父节点的每个分量施加Logistic

    def get_jacobi(self, parent):
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


class ReLU(Node):
    """
    对矩阵的元素施加ReLU函数
    """

    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value > 0.0, self.parents[0].value,
                                     0.1 * self.parents[0].value))  # 对父节点的每个分量施加 logistic

    def get_jacobi(self, parent):
        return np.diag(np.where(self.parents[0].value.A1 > 0.0, 1.0, 0.1))


class Vectorize(Node):
    """
    将多个父节点组装成一个向量
    """

    def compute(self):
        assert len(self.parents) > 0
        self.value = np.mat(np.array([node.value for node in self.parents])).T  # 将本节点的父节点的值列成向量

    def get_jacobi(self, parent):
        return np.mat([node is parent for node in self.parents]).astype(np.float).T


class SoftMax(Node):
    """
    SoftMax函数
    """

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2  # 防止指数过大
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        我们不实现SoftMax节点的get_jacobi函数，训练时使用CrossEntropyWithSoftMax节点（见下）
        """
        return np.mat(np.eye(self.dimension()))  # 无用


class CrossEntropyWithSoftMax(Node):
    """
    对第一个父节点施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
    """

    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(-np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).ravel()
        else:
            raise NotImplementedError("不会需要对label求雅克比")


class Reshape(Node):
    """
    改变父节点的值（矩阵）的形状
    """

    def __init__(self, parent, shape):
        Node.__init__(self, parent)

        assert isinstance(shape, tuple) and len(shape) == 2
        self.to_shape = shape

    def compute(self):
        self.value = self.parents[0].value.reshape(self.to_shape)

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))


class Multiply(Node):
    """
    两个父节点的值是相同形状的矩阵，将它们对应位置的值相乘
    """

    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)


class Convolve(Node):
    """
    以第二个父节点的值为卷积核，对第一个父节点的值做二维离散卷积
    """

    def __init__(self, *parents):
        assert len(parents) == 2
        Node.__init__(self, *parents)

        self.padded = None

    def compute(self):

        data = self.parents[0].value  # 输入特征图
        kernel = self.parents[1].value  # 卷积核

        w, h = data.shape  # 输入特征图的宽和高
        kw, kh = kernel.shape  # 卷积核尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 卷积核长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[hkw:hkw + w, hkh:hkh + h] = data

        self.value = np.mat(np.zeros((w, h)))

        # 二维离散卷积
        for i in np.arange(hkw, hkw + w):
            for j in np.arange(hkh, hkh + h):
                self.value[i - hkw, j - hkh] = np.sum(
                    np.multiply(self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh], kernel))

    def get_jacobi(self, parent):

        data = self.parents[0].value  # 输入特征图
        kernel = self.parents[1].value  # 卷积核

        w, h = data.shape  # 输入特征图的宽和高
        kw, kh = kernel.shape  # 卷积核尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 卷积核长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        jacobi = []
        if parent is self.parents[0]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh] = kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    jacobi.append(self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh].A1)
        else:
            raise Exception("You're not my father")

        return np.mat(jacobi)


class MaxPooling(Node):
    """
    最大值池化
    """

    def __init__(self, parent, size, stride):
        Node.__init__(self, parent)

        assert isinstance(stride, tuple) and len(stride) == 2
        self.stride = stride

        assert isinstance(size, tuple) and len(size) == 2
        self.size = size

        self.flag = None

    def compute(self):
        data = self.parents[0].value  # 输入特征图
        w, h = data.shape  # 输入特征图的宽和高
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size  # 池化核尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 池化核长宽的一半

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口中的最大值
                top, bottom = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, j - hkh), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                row.append(
                    np.max(window)
                )

                # 记录最大值在原特征图中的位置
                pos = np.argmax(window)
                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)

            result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):

        assert parent is self.parents[0] and self.jacobi is not None
        return self.flag


class Flatten(Node):
    """
    将多个父节点的值连接成向量
    """

    def compute(self):
        assert len(self.parents) > 0

        # 将所有负矩阵展平并连接成一个向量
        self.value = np.concatenate(
            [p.value.flatten() for p in self.parents],
            axis=1
        ).T

    def get_jacobi(self, parent):
        assert parent in self.parents

        dimensions = [p.dimension() for p in self.parents]  # 各个父节点的元素数量
        pos = self.parents.index(parent)  # 当前是第几个父节点
        dimension = parent.dimension()  # 当前父节点的元素数量

        assert dimension == dimensions[pos]

        jacobi = np.mat(np.zeros((self.dimension(), dimension)))
        start_row = int(np.sum(dimensions[:pos]))
        jacobi[start_row:start_row + dimension, 0:dimension] = np.eye(dimension)

        return jacobi


class ScalarMultiply(Node):
    """
    用标量（1x1矩阵）数乘一个矩阵
    """

    def compute(self):
        assert self.parents[0].shape() == (1, 1)  # 第一个父节点是标量
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        assert parent in self.parents

        if parent is self.parents[0]:
            return self.parents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.parents[1].dimension())) * self.parents[0].value[0, 0]