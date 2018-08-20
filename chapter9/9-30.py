# -*- coding:utf-8 -*-
import random
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def do_generate_x_y(isTrain, batch_size, seqlen):
    batch_x = []
    batch_y = []

    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1
        sin_data = amp_rand * np.sin(
            np.linspace(
                seqlen / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
                seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
                seqlen * 2,
            )
        )
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2
        sig_data = (
            amp_rand
            * np.cos(
                np.linspace(
                    seqlen / 15.0 * freq_rand * 0.0 * math.pi + offset_rand,
                    seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand,
                    seqlen * 2,
                )
            )
            + sin_data
        )

        batch_x.append(np.array([sig_data[:seqlen]]).T)
        batch_y.append(np.array(sig_data[seqlen:]).T)
    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))
    return batch_x, batch_y


def generate_data(isTrain, batch_size):
    seq_length = 15
    if isTrain:
        return do_generate_x_y(isTrain, batch_size, seqlen=seq_length)
    else:
        return do_generate_x_y(isTrain, batch_size, seqlen=seq_length * 2)


sample_now, sample_f = generate_data(True, 3)

seq_length = sample_now.shape[0]
batch_size = 10

output_dim = input_dim = sample_now.shape[-1]
hidden_dim = 12
layers_stacked_count = 2

learning_rate = 0.04
nb_iters = 100

lambda_l2_reg = 0.003

tf.reset_default_graph()

encoder_input = []
expected_output = []
decode_input = []
for i in range(seq_length):
    encoder_input.append(tf.placeholder(tf.float32, shape=(None, input_dim)))
    expected_output.append(tf.placeholder(tf.float32, shape=(None, output_dim)))
    decode_input.append(tf.placeholder(tf.float32, shape=(None, input_dim)))
tcells = []
for i in range(layers_stacked_count):
    tcells.append(tf.contrib.rnn.GRUCell(hidden_dim))
Mcell = tf.contrib.rnn.MultiRNNCell(tcells)

dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
    encoder_input, decode_input, Mcell
)

reshaped_outputs = []
for ii in dec_outputs:
    reshaped_outputs.append(
        tf.contrib.layers.fully_connected(ii, output_dim, activation_fn=None)
    )

output_loss = 0

for _y, _Y in zip(reshaped_outputs, expected_output):
    output_loss += tf.reduce_mean(tf.pow(_y - _Y), 2)

reg_loss = 0
for tf_var in tf.trainable_variables():
    if not ("fully_connected" in tf_var.name):
        reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
loss = output_loss + lambda_l2_reg * reg_loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.InteractiveSession()


def train_batch(batch_size):
    X, Y = generate_data(True, batch_size)
    feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})
    c = np.concatenate(([np.zeros_like(Y[0])], Y[:-1]), axis=0)
    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


train_losses = []
test_losses = []

sess.run(tf.global_variables_initializer())
for t in range(nb_iters + 1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)

print(f'Finally train loss:{train_losses},\t test loss:{test_losses}')
