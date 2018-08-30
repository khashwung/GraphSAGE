from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
用于采样邻居node
"""

class UniformNeighborSampler(Layer):
    """
    继承自Layer
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ''' 父类留给不同的子类去实现的一个方法
            inputs是一个tuple
        '''
        # ids为list，要批量操作
        ids, num_samples = inputs
        # 巧用embedding_lookup，把一个节点id的邻居看作是该节点的embedding，这样就相当于获取了节点邻居的id
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        # transpose为了沿着行，即邻居维度进行shuffle，所以随机就体现在这一步
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        # 且num_samples个邻居
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists
