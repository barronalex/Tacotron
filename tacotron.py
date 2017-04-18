from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import dynamic_attention_wrapper as wrapper, helper, basic_decoder, decoder
from tensorflow.contrib.rnn import *

def highway(inputs, units=128, scope='highway'):
    with tf.variable_scope(scope):
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

def CBHG(inputs, sequence_len, K=16, c=[128,128,128], gru_units=128, num_highway_layers=4, num_conv_proj=2):
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

    assert num_conv_proj == len(c) - 1
    conv_proj = conv_bank
    for layer in range(num_conv_proj):
        activation = None if layer == num_conv_proj - 1 else tf.nn.relu
        # conv projections
        conv_proj = tf.layers.conv1d(
                conv_bank,
                filters=c[layer+1],
                kernel_size=3,
                padding='same',
                activation=activation
        )
        conv_proj = tf.layers.batch_normalization(conv_proj)

    # residual connection
    conv_res = conv_proj + inputs

    # highway feature extraction
    h = conv_res
    for layer in range(num_highway_layers):
        h = highway(h, scope='highway_' + str(layer))

    # bi-GRU
    forward_gru_cell = GRUCell(gru_units)
    backward_gru_cell = GRUCell(gru_units)
    out = tf.nn.bidirectional_dynamic_rnn(forward_gru_cell, backward_gru_cell, h, sequence_length=sequence_len, dtype=tf.float32)
    out = tf.concat(out[0], -1)

    return out

def pre_net(inputs, units=[256,128], train=True):
    layer_1 = tf.layers.dense(inputs, units[0], activation=tf.nn.relu)
    layer_1 = tf.layers.dropout(layer_1, rate=0.5, training=train)
    layer_2 = tf.layers.dense(layer_1, units[1], activation=tf.nn.relu)
    layer_2 = tf.layers.dropout(layer_2, rate=0.5, training=train)
    return layer_2

class Tacotron(object):
    def inference(self, inputs, train=False):
        pre_out = pre_net(inputs['text'])

        encoded = CBHG(pre_out, inputs['text_len'])

        attention_mech = wrapper.BahdanauAttention(256, encoded, memory_sequence_length=inputs['text_len'])
        decoder_cell = ResidualWrapper(
                MultiRNNCell([GRUCell(384) for _ in range(3)])
        )

        def decoder_frame_input(inputs, attention):
            processed_inputs = pre_net(inputs)
            return tf.concat([processed_inputs, attention], -1)

        cell = wrapper.DynamicAttentionWrapper(
                decoder_cell,
                attention_mech,
                attention_size=256,
                cell_input_fn=decoder_frame_input,
                output_attention=False
        )

        if train:
            decoder_helper = helper.TrainingHelper(inputs['mel'], inputs['mel_sl'])
        else:
            decoder_helper = helper.GreedyEmbeddingHelper(tf.identity, tf.zeros(64, dtype=tf.int32), 0)

        dec = basic_decoder.BasicDecoder(
                cell,
                decoder_helper,
                cell.zero_state(dtype=tf.float32, batch_size=64)
        )
        outputs, _ = decoder.dynamic_decode(dec)
        outputs = tf.layers.dense(
                outputs[0],
                units=80,
                maximum_iterations=400
        )
        print(outputs.shape)

        return outputs


    def __init__(self, config, inputs):
        self.config = config
        self.output = self.inference(inputs)

class Config(object):
    dropout = 0.5


# tests
with tf.Session() as sess:
    text = tf.random_normal([64, 129, 1])
    sl = tf.ones([64], dtype=tf.int32)*50
    mel = tf.random_normal([64, 193, 80])
    mel_sl = tf.ones([64], dtype=tf.int32)*50

    config = Config()
    inputs = {'text': text, 'text_len': sl, 'mel': mel, 'mel_sl': mel_sl}
    model = Tacotron(config, inputs)

    tf.global_variables_initializer().run()
    #print(sess.run(model.output))
    





