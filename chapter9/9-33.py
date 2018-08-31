# -*- coding:utf-8 -*-
import os
import collections
import tensorflow as tf
import jieba
import re

jieba.load_userdict('myjiebdadict.txt')


def fenci(training_data):
    seg_list = jieba.cut(training_data)
    training_ci = " ".join(seg_list)
    training_ci = training_ci.split()
    return training_ci


_PAD = "_PAD"  # 用于在桶机制中为对齐填充占位
_GO = "_GO"  # 是解码输入时的开头标志位
_EOS = "_EOS"  # 用于标识输出结果的结尾
_UNK = "_UNK"  # 用来代替处理样本是出现字典里没有的字符
_NUM = "_NUM"  # 代替数字

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Isch=True中文  False英文
import numpy as np
from numpy.random import shuffle


def get_ch_path_text(raw_data_dir, Isch=True, normalize_digits=False):
    text_files, _ = getRawFileList(raw_data_dir)
    labels = []
    traning_dataszs = list([0])
    if len(text_files) == 0:
        return labels
    shuffle(text_files)
    for text_file in text_files:
        training_data, training_datasz = get_ch_label(text_file, Isch, normalize_digits)
        training_ci = np.array(training_data)
        training_ci = np.reshape(training_ci, [-1, 1])
        labels.append(training_ci)
        training_datasz = np.array(training_datasz) + traning_dataszs[-1]
        traning_dataszs.extend(list(training_datasz))

    return labels, traning_dataszs


def build_dataset(words, n_words):
    count = [[_PAD, -1], [_GO, -1], [_EOS, -1], [_UNK, -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def create_vocabulary(
    vocabulary_file, raw_data_dir, max_vocabulary_size, Isch=True, normalize_digits=True
):
    texts, textssz = get_ch_path_text(raw_data_dir, Isch, normalize_digits)
    all_words = []
    for label in texts:
        all_words += [word for word in label]
    training_label, count, dictionary, reverse_dictionary = build_dataset(
        all_words, max_vocabulary_size
    )
    if not tf.gfile.Exists(vocabulary_file):
        if len(reverse_dictionary) > max_vocabulary_size:
            reverse_dictionary = reverse_dictionary[:max_vocabulary_size]
        with tf.gfile.GFile(vocabulary_file, mode='w') as vocab_file:
            for w in reverse_dictionary:
                vocab_file.write(reverse_dictionary[w] + '\n')
    else:
        pass
    return training_label, count, dictionary, reverse_dictionary, textssz


vocab_size = 40000
raw_data_dir = ''
raw_data_dir_to = ''
data_dir = ''
vocabulary_fileen = ''
vocabulary_filech = ''


def initialize_vocabulary(vocabulary_path):
    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []
        with tf.gfile.GFile(vocabulary_path, mode='r') as f:
            rev_vocab.extend(f.readline())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (x, y) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        pass


def getRawFileList(path):
    files = []
    names = []
    for f in os.listdir(path):
        if not f.endswith('~') or not f == "":
            files.append(os.path.join(path, f))
            names.append(f)
    return files, names


def basic_tokenizer(sentence):
    _WORD_SPLIT = "([.,!?\"':;)()])"
    _CHWORD_SPLIT = "、|。，‘’"
    str1 = ""
    for i in re.split(_CHWORD_SPLIT, sentence):
        str1 = str1 + i
    str2 = ""
    for i in re.split(_WORD_SPLIT, str1):
        str2 = str2 + i
    return str2


def sentence_to_ids(sentence, vocabulary, normalize_digits=True, Isch=True):
    if normalize_digits:
        sentence = re.sub('\d+', _NUM, sentence)
    notoken = basic_tokenizer(sentence)
    if Isch:
        notoken = fenci(notoken)
    else:
        notoken = notoken.split()
    idsdata = [vocabulary.get(w, UNK_ID) for w in notoken]
    return idsdata


def textfile_to_idsfile(
    data_file_name, target_file_name, vocab, Isch=True, normalize_digits=True
):
    if not tf.gfile.Exists(target_file_name):
        with tf.gfile.GFile(data_file_name, mode='rb') as data_file:
            with tf.gfile.GFile(target_file_name, mode='w') as ids_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        pass
                    token_ids = sentence_to_ids(line, vocab, normalize_digits, Isch)
                    ids_file.write(" ".join([str(tok) for tok in token_ids]) + '\n')


def ids2texts(indices, rev_vocab):
    texts = []
    for index in indices:
        texts.append(rev_vocab[index])
    return texts


def get_ch_label(txt_file, Isch=True, normalize_digits=False):
    labels = list()
    labelssz = []
    with open(txt_file, 'rb') as f:
        for label in f:
            if normalize_digits:
                label = re.sub('\d+', _NUM, label)
            notoken = basic_tokenizer(label)
            if Isch:
                notoken = fenci(notoken)
            else:
                notoken = notoken.split()
            labels.extend(notoken)
            labelssz.append(len(labels))
    return labels, labelssz


def textdir_to_idsdir(textdir, idsdir, vocab, normalize_digits=True, Isch=False):
    text_file, filenames = getRawFileList(textdir)
    if len(text_file) == 0:
        raise ''
    for text_file, name in zip(text_file, filenames):
        textfile_to_idsfile(text_file, idsdir + name, vocab, normalize_digits, Isch)


target_train_file_path = ''

import sys

plot_histograms = True


def analysisfile(source_file, target_file):
    source_lengths = []
    target_lengths = []
    with tf.gfile.GFile(source_file) as s_file:
        with tf.gfile.GFile(target_file) as t_file:
            source = s_file.readline()
            target = t_file.readline()
            counter = 0

            while source and target:
                counter += 1
                if counter % 100000 == 0:
                    sys.stdout.flush()
                num_source_ids = len(source.split())
                source_lengths.append(num_source_ids)
                num_target_ids = len(target.split()) + 1
                target_lengths.append(num_target_ids)
                source, target = s_file.readline(), t_file.readline()
    if plot_histograms:
        pass


def main():
    vocabulary_filenameen = os.path.join(data_dir, vocabulary_fileen)
    vocabulary_filenamech = os.path.join(data_dir, vocabulary_filech)
    training_dataen, counten, dictionaryen, reverse_dictionaryen, textsszen = create_vocabulary(
        vocabulary_fileen, raw_data_dir, vocab_size, Isch=False, normalize_digits=True
    )

    training_datach, countch, dictionarych, reverse_dictionarych, textsszch = create_vocabulary(
        vocabulary_filech, raw_data_dir_to, vocab_size, Isch=True, normalize_digits=True
    )
    filefrom, _ = getRawFileList(data_dir + 'fromids/')
    filesto, _ = getRawFileList(data_dir + 'toids/')
    source_train_file_path = filefrom[0]
    source_train_file_path = filesto[0]
    analysisfile(source_train_file_path, target_train_file_path)

