# -*- coding:utf-8 -*-
import tensorflow as tf
import time
import numpy as np

n_input = 0
n_context = 0

input_tensor = tf.placeholder(
    tf.float32, [None, None, n_input+(2*n_input*n_context)], name='input')
targets = tf.sparse_placeholder(tf.int32, name='targets')
seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
keep_dropout = tf.placeholder(tf.float32)

b_stddev = 0.046875
h_stddev = 0.046875

n_hidden = 1024
n_hidden_1 = 1024
n_hidden_2 = 1024
n_hidden_5 = 1024
n_cell_dim = 1024
n_hidden_3 = 2*1024

keep_dropout_rate = 0.95
relu_clip = 20


def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def BiRNN_model(batch_x, seq_length, n_input, n_context, n_character, keep_dropout):
    batch_x_shape = tf.shape(batch_x)
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    batch_x = tf.reshape(batch_x, [-1, n_input+2*n_input*n_context])
    with tf.name_scope('fc1'):
        b1 = variable_on_cpu(
            'b1', [n_hidden_1], tf.random_normal_initializer(stddev=b_stddev))
        h1 = variable_on_cpu('h1', [n_input+2*n_input*n_context, n_hidden_1],
                             initializer=tf.random_normal_initializer(stddev=h_stddev))
        layer_1 = tf.minimum(tf.nn.relu(
            tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
        layer_1 = tf.nn.dropout(layer_1, keep_dropout)
    with tf.name_scope('fc2'):
        b2 = variable_on_cpu(
            'b2', [n_hidden_2], initializer=tf.random_normal_initializer(stddev=b_stddev))
        h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2],
                             initializer=tf.random_normal_initializer(stddev=h_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)))
        layer_2 = tf.nn.dropout(layer_2, keep_dropout)
    with tf.name_scope('fc3'):
        b3 = variable_on_cpu(
            'b3', [n_hidden_3], initializer=tf.random_normal_initializer(stddev=b_stddev))
        h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3],
                             initializer=tf.random_normal_initializer(stddev=h_stddev))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)))
        layer_3 = tf.nn.dropout(layer_3, keep_dropout)
    with tf.name_scope('lstm'):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
            n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
            n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=layer_3, dtype=tf.float32, time_major=True, sequence_length=seq_length)
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])
    with tf.name_scope('fc5'):
        b5 = variable_on_cpu(
            'b5', [n_hidden_5], initializer=tf.random_normal_initializer(stddev=b_stddev))
        h5 = variable_on_cpu('h5', [(2*n_cell_dim), n_hidden_5],
                             initializer=tf.random_normal_initializer(stddev=h_stddev))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)))
        layer_5 = tf.nn.dropout(layer_5, keep_dropout)
    with tf.name_scope('fc6'):
        b6 = variable_on_cpu(
            'b6', [n_character], initializer=tf.random_normal_initializer(stddev=b_stddev))
        h6 = variable_on_cpu('h6', [n_hidden_5, n_character],
                             initializer=tf.random_normal_initializer(stddev=h_stddev))
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_character])
    return layer_6


def ctc_loss(targets, logits, seq_length):
    pass


def ctc_beam_search_decoder(logits, seq_length, merge_repeated=False):
    return [], 0


words_size = 10000
logits = BiRNN_model(input_tensor, tf.to_int64(seq_length),
                     n_input, n_context, words_size+1, keep_dropout)
avg_loss = tf.reduce_mean(ctc_loss(targets, logits, seq_length))
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(avg_loss)
with tf.name_scope('decode'):
    decoded, log_prob = ctc_beam_search_decoder(
        logits, seq_length, merge_repeated=False)
with tf.name_scope('accuracy'):
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), targets)
    ler = tf.reduce_mean(distance, name='label_error_rate')

epochs = 100
savedir = 'save'
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    kpt = tf.train.latest_checkpoint(savedir)
    print('kpt:', kpt)
    startepo = 0
    if kpt != None:
        saver.restore(sess, kpt)
        ind = kpt.find('-')
        startepo = int(kpt[ind+1:])
        print(startepo)
        section = '\n{0:=^40}\n'
        print(section.format('Run training epoch'))
        train_start = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            if epoch < startepo:
                continue
            print('epoch start:', epoch, 'total epochs=', epochs)
            n_batch_per_epoch = int(np.ceil(len(labels)))
            print('total loop', n_batches_per_epoch,
                  'in one epoch,', batch_size, 'items in one loop')

            train_cost = 0
            train_ler = 0
            next_idx = 0

            for batch in range(n_batches_per_epoch):
                next_idx, source, source_lengths, sparse_labels = next_batch(
                    labels, next_idx, batch_size)
                feed = {input_tensor: source, targets: sparse_labels,
                        seq_length: source_lengths, keep_dropout: keep_dropout}
                batch_cost, _ = sess.run([avg_loss, optimizer], feed_dict=feed)
                train_cost += batch_cost
                if (batch+1) % 20 == 0:
                    print('loop:', batch, 'Train cost:', train_cost/(batch+1))
                    feed2 = {input_tensor: source, targets: sparse_labels,
                             seq_length: source_lengths, keep_dropout: 1.0}
                    d, train_ler = sess.run([decoded[0], ler], feed_dict=feed2)
                    dense_decoded = tf.sparse_tensor_to_dense(
                        d, default_value=-1).eval(session=sess)
                    dense_labels = sparse_tuple_to_texts_ch(
                        sparse_labels, words)
                    counter = 0
                    print('Label err rate:', train_ler)
                    for orig, decoded_arr in zip(dense_labels, dense_decoded):
                        decoded_str = ndarray_to_text_ch(decoded_arr, words)
                        counter = counter+1
                        break
                saver.save(sess, savedir+'yuyinn.cpkt', global_step=epoch)
