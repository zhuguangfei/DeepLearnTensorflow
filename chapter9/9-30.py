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

