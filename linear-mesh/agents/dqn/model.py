import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np



class QNetworkTf():
    """Actor (Policy) Model."""

    def __init__(self, session, state_size, action_size, name, learning_rate, checkpoint_file=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            name (str): Prefix for tensor names
            learning_rate (float): Network learning rate
        """
        self.sess = session
        self.name = name
        self.action_size = action_size
        self.learning_rate = learning_rate

        if checkpoint_file is None:
            with tf.variable_scope("placeholders_"+self.name):
                self.input = tf.placeholder(tf.float32, shape=(4, None, 2), name='input')
                self.y_input = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')
                self.gather_index = tf.placeholder(tf.int32, shape=(None), name='gather_index')

            self.output = self._inference()
            self.loss, self.optimizer = self._training_graph()

            self.sess.run([tf.global_variables_initializer(),
                           tf.local_variables_initializer()])
        else:
            checkpoint_dir = '/'.join(checkpoint_file.split('/')[:-1])
            saver = tf.train.import_meta_graph(checkpoint_file+'.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))

            self.input = tf.get_default_graph().get_tensor_by_name('placeholders_'+self.name+'/input:0')
            self.y_input = tf.get_default_graph().get_tensor_by_name('placeholders_'+self.name+'/y_input:0')
            self.gather_index = tf.get_default_graph().get_tensor_by_name('placeholders_'+self.name+'/gather_index:0')
            self.loss = tf.get_default_graph().get_tensor_by_name(f'training_{self.name}/loss:0')
            self.optimizer = tf.get_default_graph().get_operation_by_name(f'training_{self.name}/optimize')
            self.output = tf.get_default_graph().get_tensor_by_name(f'inference_{self.name}/dense_2/BiasAdd:0')

        self.step = 0

    def _inference(self):
        with tf.variable_scope("inference_"+self.name):
            inp = tf.unstack(self.input, axis=0)
            layer, _ = tf.contrib.rnn.static_rnn(tf.contrib.rnn.LSTMCell(8,activation=tf.nn.relu), inp, dtype=tf.float32)
            layer = tf.layers.dense(layer[-1], 128, activation=tf.nn.relu)
            #layer = tf.layers.dense(layer, 256, activation=tf.nn.relu)
            #layer = tf.layers.dropout(layer, 0.5)
            layer = tf.layers.dense(layer, 64, activation=tf.nn.relu)
            # layer = tf.layers.dense(layer, 16, activation=tf.nn.relu)
            output = tf.layers.dense(layer, self.action_size)
        return output

    def _training_graph(self):
        with tf.variable_scope('training_'+self.name):
            pad = tf.range(tf.size(self.gather_index))
            pad = tf.expand_dims(pad, 1)
            ind = tf.concat([pad, self.gather_index], axis=1)

            gathered = tf.gather_nd(self.output, ind)
            gathered = tf.expand_dims(gathered, 1)
            loss = tf.losses.mean_squared_error(
                labels=self.y_input, predictions=gathered)
            # loss = tf.multiply(self.loss_modifier, loss)
            loss = tf.reduce_mean(loss, name='loss')

            # decayed_lr = tf.train.exponential_decay(5e-4,
            #                             tf.train.get_or_create_global_step(), 10000,
            #                             0.95, staircase=True)
            optimize = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(loss, name='optimize')

        return loss, optimize

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.sess.run(self.output, feed_dict={self.input: state})

    def train(self, states, y_correct, actions):
        reduced, result, _ = self.sess.run([self.loss, self.output, self.optimizer], feed_dict={
            self.input: states, self.y_input: y_correct, self.gather_index: actions})
        return reduced, result

