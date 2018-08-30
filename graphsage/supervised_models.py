import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE.
        有监督式的实现，还有一份无监督的实现，可以看到有监督的这个实现是继承自无监督的实现
    """

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples) 采样也是一种pad方法
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        # 类似于策略模式。用户传入不同的参数，可以使用不同的aggregate方法
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           # identity_dim一般设为[64,256]，该flag用于embed节点id，提高参数量的同时也提高精度，这也是要学习的参数
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None:
            # 如果没有特征，identity_dim必须要指定，作为模型的特征
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                # 节点embed和节点本来的特征通过concat堆叠在一起
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        # 总共的特征维度
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        # 显然self.dims是要存下每个layer的输出维度，包括一开始的输入维度和最终的输出维度
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()


    def build(self):
        # sample1是被摊平的，但是通过support size可以reshape回来
        # 这里有个思想和计算一阶二阶邻居不一样，就是这里的邻居是随机采样邻居！！因此可以将邻居的数量normalize一下，这样做的好处
        # 有两个：1.所有节点的邻居数量等长，计算方便；2.邻居数量可控，避免某些超级节点导致的内存消耗非常巨大的问题
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        dim_mult = 2 if self.concat else 1

        # 最后一层没有经过relu，所以归一化一下
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        # 最后初始化一个dense层做预测
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        # 进行预测
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # 梯度clipping，防止梯度爆炸
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        # 因为最后返回的self.node_preds是logits，因此这里再经过一个softmax，事实上计算损失的时候已经经过softmax了
        self.preds = self.predict()

    def _loss(self):
        # Weight decay loss
        # aggregator和最后一层dense是分开声明的，所以只能分别weight decay，最后损失加在一起
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        # multi label也可以使用cross entropy损失，因为本质上cross entropy描述的是gt与pred直接的分布差异
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
