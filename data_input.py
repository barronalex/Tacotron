from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import pickle as pkl
import os
import sys
import io

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 10000

def build_dataset(sess, inputs, names):
    placeholders = []
    for inp in inputs:
        placeholders.append(tf.placeholder(inp.dtype, inp.shape))

    with tf.device('/cpu:0'):
        dataset = tf.contrib.data.Dataset.from_tensor_slices(placeholders)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_initializable_iterator()

        batch_inputs = iterator.get_next()
        batch_inputs = {na: inp for na, inp in zip(names, batch_inputs)}
        for name, inp in batch_inputs.items():
            print(name, inp)

        sess.run(iterator.initializer, feed_dict=dict(zip(placeholders, inputs)))
        batch_inputs['stft'] = tf.cast(batch_inputs['stft'], tf.float32)
        batch_inputs['mel'] = tf.cast(batch_inputs['mel'], tf.float32)

    return batch_inputs

def load_from_npy(dirname):
    text = np.load(dirname + 'texts.npy')
    text_length = np.load(dirname + 'text_lens.npy')
    print('loading stft')
    stft = np.load(dirname + 'stfts.npy')
    print('loading mel')
    mel = np.load(dirname + 'mels.npy')
    speech_length = np.load(dirname + 'speech_lens.npy')

    print('normalizing')
    # normalize
    # take a sample to avoid memory errors
    index = np.random.randint(len(stft), size=100)

    stft_mean = np.mean(stft[index], axis=(0,1))
    mel_mean = np.mean(mel[index], axis=(0,1))
    stft_std = np.std(stft[index], axis=(0,1), dtype=np.float32)
    mel_std = np.std(mel[index], axis=(0,1), dtype=np.float32)

    stft -= stft_mean
    mel -= mel_mean
    stft /= stft_std
    mel /= mel_std

    text = np.array(text, dtype=np.int32)
    text_length = np.array(text_length, dtype=np.int32)
    speech_length = np.array(speech_length, dtype=np.int32)

    # NOTE: reconstruct zero frames as paper suggests
    speech_length = np.ones(text.shape[0], dtype=np.int32)*mel.shape[1]

    inputs = list((text, text_length, stft, mel, speech_length))
    names = ['text', 'text_length', 'stft', 'mel', 'speech_length']

    if os.path.exists(dirname + 'speakers.npy'):
        speakers = np.load(dirname + 'speakers.npy')
        inputs.append(speakers)
        names.append('speaker')

    return inputs, names, stft_mean, stft_std

def pad(text, max_len, pad_val):
    return np.array(
        [np.pad(t, (0, max_len - len(t)), 'constant', constant_values=pad_val) for t in text]
    , dtype=np.int32)

def load_prompts(prompt_file, ivocab):
    vocab = {v: k for k,v in ivocab.items()}
    with open(prompt_file, 'r') as pf:
        lines = pf.readlines() 
        text = [[vocab[w] for w in l.strip() if w in vocab] for l in lines]
        text_length = np.array([len(l) for l in lines])
        text = pad(text, np.max(text_length), 0)
        
        inputs = tf.train.slice_input_producer([text, text_length], num_epochs=1)
        inputs = {'text': inputs[0], 'text_length': inputs[1]}

        batches = tf.train.batch(inputs,
                batch_size=32,
                allow_smaller_final_batch=True)
        print(batches)
        return batches, len(lines)
        
def load_meta(data_path):
    with open('%smeta.pkl' % data_path, 'rb') as vf:
        meta = pkl.load(vf)
    return meta

def generate_attention_plot(alignments):
    plt.imshow(alignments, cmap='hot', interpolation='nearest')
    plt.ylabel('Decoder Steps')
    plt.xlabel('Encoder Steps')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot = tf.image.decode_png(buf.getvalue(), channels=4)
    plot = tf.expand_dims(plot, 0)
    return plot

