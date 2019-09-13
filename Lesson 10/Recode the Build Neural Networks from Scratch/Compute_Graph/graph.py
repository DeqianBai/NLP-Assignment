#！/usr/bin/env python
#  -*- coding:utf-8 -*-
#  author:dabai time:2019/9/12

import os

class Graph:
    """
    计算图类
    """

    def __init__(self):
        self.nodes = [] # 计算图内的节点列表

    def add_node(self,node):
        """
        添加节点
        :param node:
        :return:
        """
        self.nodes.append(node)

    def clear_jacobi(self):
        """
        清除图中全部节点的雅可比矩阵
        :return:
        """
        for node in self.nodes:
            node.clear_jacobi()

    def reset_value(self):
        """
        重置图中全部节点的值
        :return:
        """
        for node in self.nodes:
            node.reset_value(False)

    def draw(self,ax=None):
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            import numpy as np

        except:
            raise Exception("Need Module networkx")

        G = nx.Graph()

        already = []
        labels = {}

        for node in self.nodes:
            G.add_node(node)
            labels[node] = node.__class__.__name__ + ("{:s}".format(str(node.dim)) if  hasattr(node,"dim") else "")\
                           + ("\n[{:.3f}]".format(np.linalg.norm(node.jacobi)) if node.jacobi is not None else "")

            for c in node.get_children():
                if {node,c} not in already:
                    G.add_edge(node,c)
                    already.append({node,c})

            for p in node.get_parents():
                if {node,p} not in already:
                    G.add_edge(node,p)
                    already.append({node,p})

        savefig = False
        if ax is None:
            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(111)
            savefig = True


        ax.clear()
        ax.axis("on")
        ax.grid(True)

        pos = nx.spring_layout(G)

        # 有雅可比的变量节点
        cm = plt.cm.Reds
        nodelist = [n for n in self.nodes if n.__class__.__name__ == "Variable" and n.jacobi is not None]
        colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist,cmap=cm,edgecolors="#666666",
                               node_size=2000,alpha=1.0,ax=ax)

        # 无雅可比的变量节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ == "Variable" and n.jacobi is None]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)
        # 有雅可比的计算节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ != "Variable" and n.jacobi is not None]
        colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0, ax=ax)

        # 无雅可比的中间节点
        nodelist = [n for n in self.nodes if n.__class__.__name__ != "Variable" and n.jacobi is None]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", cmap=cm, edgecolors="#666666",
                               node_size=2000, alpha=1.0,ax=ax)

        # 边
        nx.draw_networkx_edges(G, pos, width=2, edge_color="#014b66", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, labels=labels, font_weight="bold", font_color="#6c6c6c", font_size=8,
                                     font_family='arial', ax=ax)


        # 保存图像
        if savefig:
            file = "./picture/computing_graph.png"
            if os.path.exists(file):
                os.remove(file)

            plt.savefig(file)
            plt.close()


# 全局默认计算图
# default_graph是一个全局的Graph对象，默认情况下所有节点都将被加入到default_graph中
default_graph = Graph()