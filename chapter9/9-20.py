from tensorflow.contrib.rnn.ops import gen_gru_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.platform import resource_loader
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.ops.math_ops import sigmoid
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear

LayerRNNCell = rnn_cell_impl.LayerRNNCell


print(tf.__version__)
tf.reset_default_graph()


def ln(tensor, scope=None, epsilon=1e-5):
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable(
            'scale',
            shape=[tensor.get_shape()[1]],
            initializer=tf.constant_initializer(1),
        )
        shift = tf.get_variable(
            'shift',
            shape=[tensor.get_shape()[1]],
            initializer=tf.constant_initializer(0),
        )
    LN_inital = (tensor - m) / tf.sqrt(v + epsilon)
    return LN_inital * scale + shift


class LNGRUBlockCell(LayerRNNCell):
    def __init__(self, num_units=None, input_size=None, activation=tf.tanh):
        if input_size is not None:
            print('%s: The input_size parameter is deprecated.' % self)
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state):
        with vs.variable_scope('Gates'):
            value = _linear([inputs, state], 2 * self._num_units, True, 1.0)
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
            r = ln(r, scope='r/')
            u = ln(u, scope='u/')
            r, u = sigmoid(r), sigmoid(u)
        with vs.variable_scope('Candidate'):
            Cand = _linear([inputs, r * state], self._num_units, True)
            c_pre = ln(Cand, scope='new_h/')
            c = self._activation(c_pre)
        new_h = u * state + (1 - u) * c
        return new_h, new_h

