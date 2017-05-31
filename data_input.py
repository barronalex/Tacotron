from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import pickle as pkl
import os
import sys

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

def build_dataset(sess, inputs):
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
        names = ['text', 'text_length', 'stft', 'mel', 'speech_length']
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
    index = np.random.randint(len(stft), size=1000)
    stft_mean = np.mean(stft[index], axis=(0,1))
    mel_mean = np.mean(mel[index], axis=(0,1))
    stft_std = np.std(stft[index], axis=(0,1))
    mel_std = np.std(mel[index], axis=(0,1))

    np.save(dirname + 'stft_mean', stft_mean)
    np.save(dirname + 'stft_std', stft_std)

    stft -= stft_mean
    mel -= mel_mean
    stft /= stft_std
    mel /= mel_std

    text = np.array(text, dtype=np.int32)
    text_length = np.array(text_length, dtype=np.int32)
    speech_length = np.array(speech_length, dtype=np.int32)
    mel = np.array(mel, dtype=np.float32)

    # NOTE: reconstruct zero frames as paper suggests
    speech_length = np.ones(text.shape[0], dtype=np.int32)*mel.shape[1]

    inputs = list((text, text_length, stft, mel, speech_length))
    
    return inputs, stft_mean, stft_std

def pad(text, max_len, pad_val):
    return np.array(
        [np.pad(t, (0, max_len - len(t)), 'constant', constant_values=pad_val) for t in text]
    )

def load_prompts(prompt_file, ivocab):
    vocab = {v: k for k,v in ivocab.items()}
    with open(prompt_file, 'r') as pf:
        lines = pf.readlines() 
        text = [[vocab[w] for w in l.strip()] for l in lines]
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
    with open('%s/meta.pkl' % data_path, 'rb') as vf:
        meta = pkl.load(vf)
    return meta

