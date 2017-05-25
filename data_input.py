import tensorflow as tf
import numpy as np
import pickle as pkl
import os
import threading
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

    return batch_inputs

def load_from_npy(dirname):
    text = np.load(dirname + 'texts.npy')
    text_length = np.load(dirname + 'text_lens.npy')
    stft = np.load(dirname + 'stfts.npy')
    mel = np.load(dirname + 'mels.npy')
    speech_length = np.load(dirname + 'speech_lens.npy')

    # normalize
    stft_mean = np.mean(stft, axis=(0,1))
    mel_mean = np.mean(mel, axis=(0,1))
    stft_std = np.std(stft, axis=(0,1))
    mel_std = np.std(mel, axis=(0,1))

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

def read_sequence_example(filename_queue, r=1):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context, features = tf.parse_single_sequence_example(
            serialized_example,
            context_features={
                'speech_length': tf.FixedLenFeature([], tf.int64),
                'text_length': tf.FixedLenFeature([], tf.int64),
                'speaker': tf.FixedLenFeature([], tf.int64)
                },
            sequence_features={
                'stft': tf.FixedLenSequenceFeature([], tf.float32),
                'mel': tf.FixedLenSequenceFeature([], tf.float32),
                'text': tf.FixedLenSequenceFeature([], tf.int64)
                }
            )
    # create one dictionary with all inputs
    features.update(context)

    features['stft'] = tf.reshape(features['stft'], (-1, 1025*r))
    features['mel'] = tf.reshape(features['mel'], (-1, 80*r))

    return features

def batch_inputs(example, batch_size=32, r=1):
    with tf.device('/cpu:0'):
        # separate context length so it can be used to determine bucketing
        speech_length = tf.to_int32(example['speech_length'])
        del example['speech_length']
        boundaries = [int(x/r) for x in [200, 400, 600, 800, 1000]]
        speech_length, batches = tf.contrib.training.bucket_by_sequence_length(
                    speech_length,
                    example,
                    batch_size=batch_size,
                    dynamic_pad=True,
                    bucket_boundaries=boundaries
            )
        batches['speech_length'] = speech_length
        return batches

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
        text = pad(text, np.max(text_length), vocab[' '])
        
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

# basic test
if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(['data/VCTK-Corpus/train.proto'], num_epochs=None)

    batches = batch_inputs(filename_queue)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for _ in range(20):
            print(sess.run(batches))




