import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.helper import CustomHelper
from tensorflow.contrib.rnn import *

class InferenceHelper(CustomHelper):

    def _initialize_fn(self):
        # we always reconstruct the whole output
        finished = tf.tile([False], [self._batch_size])
        next_inputs = tf.zeros([self._batch_size, self._out_size], dtype=tf.float32)
        return (finished, next_inputs)

    def _sample_fn(self, time, outputs, state):
        # we're not sampling from a vocab so we don't care about this function
        return tf.zeros(32, dtype=tf.int32)

    def _next_inputs_fn(self, time, outputs, state, sample_ids):
        del time, sample_ids
        finished = tf.tile([False], [self._batch_size])
        next_inputs = outputs
        return (finished, next_inputs, state)

    def __init__(self, batch_size, out_size):
        self._batch_size = batch_size
        self._out_size = out_size

def highway(inputs, units=128):
    # correct input shape
    if inputs.shape[-1] != units:
        inputs = tf.layers.dense(inputs, units=units)

    T = tf.layers.dense(
            inputs,
            units=units,
            activation=tf.nn.sigmoid,
    )
    # TODO update bias initial value

    H = tf.layers.dense(
            inputs,
            units=units,
            activation=tf.nn.relu
    )

    C = H*T + inputs*(1-T)
    return C

def CBHG(inputs, speaker_embed=None,
        K=16, c=[128,128,128], gru_units=128, num_highway_layers=4, num_conv_proj=2):

    with tf.variable_scope('cbhg'):

        # 1D convolution bank
        conv_bank = [tf.layers.conv1d(
            inputs,
            filters=c[0],
            kernel_size=k,
            padding='same',
            activation=tf.nn.relu
        ) for k in range(1, K+1)]

        conv_bank = tf.concat(conv_bank, -1)

        conv_bank = tf.layers.batch_normalization(conv_bank)

        conv_bank = tf.layers.max_pooling1d(
                conv_bank, 
                pool_size=2,
                strides=1,
                padding='same'
            )

        tf.summary.histogram('conv_bank', conv_bank)

        assert num_conv_proj == len(c) - 1
        conv_proj = conv_bank
        for layer in range(num_conv_proj):
            activation = None if layer == num_conv_proj - 1 else tf.nn.relu
            # conv projections
            conv_proj = tf.layers.conv1d(
                    conv_proj,
                    filters=c[layer+1],
                    kernel_size=3,
                    padding='same',
                    activation=activation
            )
            conv_proj = tf.layers.batch_normalization(conv_proj)

        tf.summary.histogram('conv_proj', conv_proj)

        # residual connection
        conv_res = conv_proj + inputs

        tf.summary.histogram('conv_res', conv_res)

        # highway feature extraction
        h = conv_res
        for layer in range(num_highway_layers):
            with tf.variable_scope('highway_' + str(layer)):

                # site specific speaker embedding
                if speaker_embed:
                    s = tf.layers.dense(speaker_embed, h.shape[-1])
                    h = tf.concat([tf.expand_dims(s), h])

                h = highway(h)

        tf.summary.histogram('highway_out', h)

        # site specfic speaker embedding
        if speaker_embed:
            s = tf.layers.dense(speaker_embed, gru_units)
        else:
            s = None

        # bi-GRU
        forward_gru_cell = GRUCell(gru_units)
        backward_gru_cell = GRUCell(gru_units)
        out, _ = tf.nn.bidirectional_dynamic_rnn(
                forward_gru_cell,
                backward_gru_cell,
                h,
                initial_state_fw=s,
                initial_state_bw=s,
                dtype=tf.float32
        )
        out = tf.concat(out, 2)

        tf.summary.histogram('encoded', out)

        return out
