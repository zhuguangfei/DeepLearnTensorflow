import tensorflow as tf
from tensorflow.contrib import rnn

learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

tf.reset_default_graph()

x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

x1 = tf.unstack(x, n_steps, 1)
lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, outputs_states = tf.nn.bidirectional_dynamic_rnn(
    lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32
)
print(len(outputs), outputs[0].shape, outputs[1].shape)
outputs = tf.concat(outputs, 2)
outputs = tf.transpose(outputs, [1, 0, 2])
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

