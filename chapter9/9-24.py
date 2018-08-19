import os
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav


wav_path = ''
label_file = ''


def get_wavs_lables(wav_path=wav_path, label_file=label_file):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_sizes < 240000:  # 剔除掉一些小文件
                    continue
                wav_files.append(filename_path)
    labels_dict = {}
    with open(label_file, 'rb') as f:
        for label in f:
            label = label.strip(b'\n')
            label_id = label.split(b' ', 1)[0]
            label_text = label.split(b' ', 1)[1]
            labels_dict[label_id.decode('ascii')] = label_text.decode('utf-8')
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)
    return new_wav_files, labels


def get_audio_transcriiptch(txt_files, wav_files, n_input, n_context, word_num_map, txt_labels=None):
    audio = []
    audio_len = []
    transcript = []
    transcript_len = []
    if txt_files != None:
        txt_labels = txt_files
    for txt_obj, wav_file in zip(txt_labels, wav_files):
        audio_data = None
        audio.append(audio_data)
        audio_len.append(np.int32(len((audio_data))))
        target = []
        if txt_files != None:
            target = None
        else:
            target = None
        transcript.append(target)
        transcript_len.append(len(target))
    audio = np.asarray(audio)
    audio_len = np.asarray(audio_len)
    transcript = np.asarray(transcript)
    transcript_len = np.asarray(transcript_len)
    return audio, audio_len, transcript, transcript_len


def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    fs, audio = wav.read(audio_filename)
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)
    orig_inputs = orig_inputs[::2]
    train_inputs = np.array([], np.float32)
    train_inputs.resize(orig_inputs.shape[0], numcep+2*numcep*numcontext)
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))
    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0]+numcontext
    context_past_max = time_slices[-1]-numcontext
    for time_slice in time_slices:
        need_empty_past = max(0, (context_past_min-time_slice))
        empty_source_past = list(
            empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(
            0, time_slice-numcontext):time_slice]
        need_empty_future = max(0, (time_slice-context_past_max))\
            empty_source_future = list(
                empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice+1:time_slice+numcontext+1]
        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past
        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future
        past = np.reshape(past, numcontext*numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext*numcep)
        train_inputs[time_slice] = np.concatenate((past, now, future))
    train_inputs = (train_inputs-np.mean(train_inputs))/np.std(train_inputs)
    return train_inputs


def pad_sequences(sequences, maxlen=None, dtype=np.float32, padding='post', truncating='post', value=0):
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    sample_shape = tuple()
    for s in sequences:
        sample_shape = np.asarray(s).shape[1:]
        break
    x = ((np.ones(nb_samples, maxlen)+sample_shape)*value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(
                'Truncating type "%s" not understood' % truncating)
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' % (
                trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


def get_ch_lable_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)

    def to_num(word): return word_num_map.get(word, words_size)
    if txt_file != None:
        txt_label = get_ch_lable(txt_file)
    labels_vector = list(map(to_num, txt_label))
    return labels_vector


def get_ch_lable(txt_file):
    labels = " "
    with open(txt_file, 'rb') as f:
        for label in f:
            labels = labels+label.decode('gb2312')
    return labels


def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend([n]*len(seq), range(len(seq)))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1]+1], dtype=np.int64)
    return indices, values, shape


SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a')-1


def sparse_tuple_to_texts_ch(tuple, words):
    indices = tuple[0]
    values = tuple[1]
    results = ['']*tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        c = ' ' if c == SPACE_INDEX else words[c]
        results[index] = results[index]+c
    return results


def ndarray_to_text_ch(value, words):
    results = ''
    for i in range(len(value)):
        results += words[value[i]]
    return results.replace(('`', ' '))
