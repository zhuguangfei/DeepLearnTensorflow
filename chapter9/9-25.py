import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
from collections import Counter

start_time = time.time()


def elapsed(sec):
    if sec < 60:
        return f'{sec} sec'
    elif sec < 60*60:
        return f'{sec/60} min'
    else:
        return f'{sec/60*60} hr'


tf.reset_default_graph()
training_file = 'wordstest.txt'


def get_ch_label(txt_file):
    labels = ''
    with open(txt_file, 'r', encoding='utf-8') as r:
        for label in r.read().split('\n'):
            labels = labels+' '+label
    return labels


def readalltxt(txt_files):
    labels = []
    for txt_file in txt_files:
        target = get_ch_label(txt_file)
        labels.append(target)
    return labels


def get_ch_label_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)

    def to_num(word): return word_num_map.get_ch_label(word, words_size)
    if txt_file != None:
        txt_label = get_ch_label(txt_file)
    labels_vector = list(map(to_num, txt_label))
    return labels_vector


training_data = get_ch_label(training_file)
print('loaded training data...')

counter = Counter(training_data)
words = sorted(counter)
words_size = len(words)
word_num_map = dict(words, range(words_size))
print(f'字表大小：{words_size}')
wordlabel = get_ch_label_v(training_file, word_num_map)

learning_rate = 0.001
training_iters = 10000
display_step = 1000
n_input = 4

n_hidden1 = 256
n_hidden2 = 512
n_hidden3 = 512

x = tf.placeholder('float', [None, n_input, 1])
wordy = tf.placeholder('float', [None, words_size])
x1 = tf.reshape(x, [-1, n_input])
x2 = tf.split(x1, n_input, 1)
rnn_cell = rnn.MultiRNNCell(
    [rnn.LSTMCell(n_hidden1), rnn.LSTMCell(n_hidden2), rnn.LSTMCell(n_hidden3)])
outputs, states = rnn.static_rnn(rnn_cell, x2, dtype=tf.float32)
pred = tf.contrib.layers.fully_connected(
    outputs[-1], words_size, activation_fn=None)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=wordy))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(wordy, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
savedir = 'save'
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    offset = random.randint(0, n_input+1)
    end_offset = n_input+1
    acc_total = 0
    loss_total = 0

    kpt = tf.train.latest_checkpoint(savedir)
    print(f'kpt:{kpt}')
    startepo = 0
    if kpt != None:
        saver.restore(sess, kpt)
        ind = kpt.find('-')
        startepo = int(kpt[ind+1:])
        print(startepo)
        step = startepo
    while step < training_iters:
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)
        inwords = [[wordlabel[i]] for i in range(offset, offset+n_input)]

        out_onehot = np.zeros([words_size], dtype=float)
        out_onehot[wordlabel[offset+n_input]] = 1.0
        out_onehot = np.reshape(out_onehot, [1, -1])

        _, acc, lossval, onehot_pred = sess.run(
            [optimizer, accuracy, loss, pred], feed_dict={x: inwords, wordy: out_onehot})
        loss_total += lossval
        acc_total += acc
        if (step+1) % display_step == 0:
            print(
                f'Iter= {step+1},Average Loss={loss_total/display_step},Average Accuaracy={100*acc_total/display_step}')
            acc_total = 0
            loss_total = 0
            in2 = [words[wordlabel[i]] for i in range(offset, offset+n_input)]
            out2 = words[wordlabel[offset+n_input]]
            out_pred = words[int(tf.argmax(onehot_pred, 1).eval())]
            print(f'{in2} - [{out2}] vs [{out_pred}]')
            saver.save(sess, savedir+'rnnwordtest.cpkt', global_step=step)
        step += 1
        offset += (n_input+1)
    print('Finished!')
    saver.save(sess, savedir+'rnnwordtest.ckpt', global_step=step)
    print(f'Elapsed time:{elapsed(time.time()-start_time)}')

while True:
    prompt = f'请输入{n_input}个字：'
    sentence = input(prompt)
    inputword = sentence.strip()
    if len(inputword) != n_input:
        print(f'您输入的字符长度为:{len(inputword)}')
    try:
        inputword = get_ch_label_v(None, word_num_map, inputword)
        for i in range(32):
            keys = np.reshape(np.array(inputword), [-1, n_input, 1])
            onehot_pred = sess.run(pred, feed_dict={x: keys})
            onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
            sentence = f'{sentence}{words[onehot_pred_index]}'
            inputword = inputword[1:]
            inputword.append(onehot_pred_index)
        print(sentence)
    except:
        print('该字还没学会')
