import tensorflow as tf

def read_sequence_example(filename_queue):
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
                'stft_real': tf.FixedLenSequenceFeature([], tf.float32),
                'stft_imag': tf.FixedLenSequenceFeature([], tf.float32),
                'mel_real': tf.FixedLenSequenceFeature([], tf.float32),
                'mel_imag': tf.FixedLenSequenceFeature([], tf.float32),
                'text': tf.FixedLenSequenceFeature([], tf.int64)
                }
            )
    # create one dictionary with all inputs
    features.update(context)

    stft = tf.complex(features['stft_real'], features['stft_imag'])
    mel = tf.complex(features['mel_real'], features['mel_imag'])

    features['stft'] = tf.reshape(stft, (1025, -1))
    features['mel'] = tf.reshape(mel, (80, -1))

    return features

def batch_inputs(filename_queue, batch_size=32):
    with tf.device('/cpu:0'):
        example = read_sequence_example(filename_queue)

filename_queue = tf.train.string_input_producer(['data/VCTK-Corpus/train.proto'], num_epochs=None)

example = read_sequence_example(filename_queue)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for _ in range(20):
        print(sess.run(example))

