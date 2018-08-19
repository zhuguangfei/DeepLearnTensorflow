import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])
x1 = tf.unstack(x, n_steps, 1)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

# 动态RNN
gru = tf.contrib.rnn.GRUCell(n_hidden)
outputs, _ = tf.nn.dynamic_rnn(gru, x, dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)
