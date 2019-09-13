#！/usr/bin/env python
#  -*- coding:utf-8 -*-
#  author:dabai time:2019/9/10
"""
http://weiweizhao.com/28_%E5%8A%A8%E6%89%8B%E5%AE%9E%E7%8E%B0tensorflow-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%ADbackpropagation/

"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import resample
import matplotlib.pyplot as plt


## 建立基类
class Node:
    """
    Each node in neural networks will have these attributes and methods

    inputs   : List,   the input nodes of this node
    outputs  : List,   the output node of this node
    gradients: Mapper, the gradient map the node of its inputs node
    forward  : Function, how to calculate the inputs
    backwards: Function, how to get the gradients when back propagation
    """
    def __init__(self,inputs=[]):
        """
        if the node is the operator of 'ax + b',the inputs will be x node,
        the outputs of this is its successors.
        the value is 'ax + b'
        :param inputs:
        """
        # A list of nodes with edges into this node.
        self.inputs = inputs # inputs_list <- c, java <-匈牙利命名法->Python 不建议这样写
        # The eventual value of this node. Set by running the forward() method.
        self.value = None

        # A list of nodes that this node outputs to.
        self.outputs = []

        # New property!
        # keys are the inputs to this node
        # and their values are the partials of this node with respect to that input.
        self.gradients = {}


        # 连接关系:输入当中每一点的输出加上它自己
        # Sets this node as an outbound node for all of this node's inputs.
        # 将此节点设置为此节点的所有输入的出节点。
        for node in self.inputs:
            node.outputs.append(self) # build a connection relationship


    def forward(self):
        # 虚类，没有实现，如果有子类，则必须重新实现这个方法
        # Every node that uses this class as a base class will need to define its own `forward` method.
        """
        Forward propagation
        compute the output value based on input nodes and store the value into "self.value"
        :return:
        """
        raise NotImplemented


    def backward(self):
        # Every node that uses this class as a base class will need to define its own `backward` method.
        """
        Back propagation
        compute the gradient of each input node and store the value into "self.gradients"
        :return:
        """
        raise NotImplemented



class Input(Node):
    """
    A generic input into the network.
    """

    def __init__(self,name=''):
        # The base class constructor has to run to set all the properties here.

        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.

        Node.__init__(self,inputs=[])
        self.name = name


    def forward(self, value = None):
        # Do nothing because nothing is calculated.
        if value is not None:
            self.value = value

    def backward(self):
        # An Input node has no inputs so the gradient(derivative) is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {}

        # Weights and bias may be inputs, so you need to sum the gradient from output gradients.
        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self] = grad_cost

    def __repr__(self):
        return 'Input Node: {}'.format(self.name)



class Linear(Node):
    """
    Represents a node that performs a linear transform.
    """
    def __init__(self,nodes,weights,bias):
        # The base class (Node) constructor. Weights and bias
        # are treated like inbound nodes.
        self.w_node = weights
        self.x_node = nodes
        self.b_node = bias
        Node.__init__(self,inputs=[nodes,weights,bias])


    def forward(self):
        """compute the wx + b using numpy"""

        self.value = np.dot(self.x_node.value, self.w_node.value) + self.b_node.value

    def backward(self):
        """ Calculates the gradient based on the output values"""
        for node in self.outputs:
            # Get the partial of the cost with respect to this node.
            # gradient_of_loss_of_this_output_node = node.gradients[self]
            grad_cost = node.gradients[self]

            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.x_node] = np.dot(grad_cost, self.w_node.value.T)

            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.w_node] = np.dot(self.x_node.value.T, grad_cost)

            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.b_node] = np.sum(grad_cost * 1, axis=0, keepdims=False)




class Sigmoid(Node):
    """
    Represents a node that performs the sigmoid activation function.
    """
    def __init__(self,node):
        # The base class constructor.
        self.x_node = node
        Node.__init__(self,[node])

    def _sigmoid(self,x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1 + np.exp(-1 * x))

    def forward(self):
        """
        Perform the sigmoid function and set the value.
        :return:
        """
        self.value = self._sigmoid(self.x_node.value)


    def backward(self):
        """
        Calculates the gradient using the derivative of the sigmoid function.
        :return:
        """
        y = self.value

        # Sigmoid对 y 的偏导
        self.partial = y * (1 - y)

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outputs:

            # loss 对Sigmoid的偏导
            grad_cost = n.gradients[self]

            self.gradients[self.x_node] = grad_cost * self.partial


class MSE(Node):
    """
    Calculates the mean squared error.
    """
    def __init__(self,y_true,y_hat):
        self.y_true_node = y_true
        self.y_hat_node = y_hat
        Node.__init__(self,inputs=[y_true,y_hat])


    def forward(self):
        # 拉平
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y_true_flatten  = self.y_true_node.value.reshape(-1,1)
        y_hat_flatten = self.y_hat_node.value.reshape(-1,1)

        # Save the computed output for backward.
        self.diff = y_true_flatten - y_hat_flatten
        self.value = np.mean(self.diff**2)

    def backward(self):
        """
        Calculates the gradient of the cost.
        This is the final node of the network so outbound nodes
        are not a concern.
        """
        n = self.y_hat_node.value.shape[0]

        self.gradients[self.y_true_node] = (2 / n) * self.diff
        self.gradients[self.y_hat_node] = (-2 / n) *self.diff


def forward_and_backward(topological_sorted_graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.
    Arguments: `graph`: The result of calling `topological_sort`.
    """
    # 输入一个图，每一个结点先forward一遍
    # 这个图是一个经过拓扑排序之后的List
    # Forward pass
    for node in topological_sorted_graph:
        node.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for node in topological_sorted_graph[::-1]:
        node.backward()


def topological_sort(data_with_value):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.
    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.
    :return:a list of sorted nodes.
    """
    feed_dict = data_with_value
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outputs:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]
            ## if n is Input Node, set n'value as
            ## feed_dict[n]
            ## else, n's value is caculate as its
            ## inbounds

        L.append(n)
        for m in n.outputs:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def sgd_update(trainable_nodes, learning_rate=1e-2):
    # there are so many other update / Optimization methods
    # such as Adam, Mom,
    for t in trainable_nodes:
        t.value += -1 * learning_rate * t.gradients[t]


