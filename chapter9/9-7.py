import tensorflow as tf

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

stacked_rnn = []
for i in range(3):
    stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
mcell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)

x1 = tf.unstack(x, n_steps, 1)
outputs, states = tf.contrib.rnn.static_rnn(mcell, x1, dtype=tf.float32)
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)
learning_rate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

training_iters = 10000
display_step = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

gru = tf.contrib.rnn.GRUCell(n_hidden * 2)
lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden)
mcell = tf.contrib.rnn.MultiRNNCell([lstm_cell, gru])

