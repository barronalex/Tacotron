import tensorflow as tf
import numpy as np
import pickle as pkl

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

def batch_inputs(filename_queue, batch_size=32, r=1, train=True):
    with tf.device('/cpu:0'):
        example = read_sequence_example(filename_queue, r=r)
        if train:
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
        else:
            batches = tf.train.batch(example,
                    batch_size=batch_size,
                    dynamic_pad=True,
                    allow_smaller_final_batch=True)
            return batches

def load_meta():
    with open('data/meta.pkl', 'rb') as vf:
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




