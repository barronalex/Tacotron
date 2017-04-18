from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import dynamic_attention_wrapper as wrapper, helper, basic_decoder, decoder
from tensorflow.contrib.rnn import *

from ops import InferenceHelper, CBHG


class Tacotron(object):
    def pre_net(self, inputs, units=[256,128], train=True):
        layer_1 = tf.layers.dense(inputs, units[0], activation=tf.nn.relu)
        layer_1 = tf.layers.dropout(layer_1, rate=0.5, training=train)
        layer_2 = tf.layers.dense(layer_1, units[1], activation=tf.nn.relu)
        layer_2 = tf.layers.dropout(layer_2, rate=0.5, training=train)
        return layer_2

    def create_decoder(self, encoded, train=True):
        attention_mech = wrapper.BahdanauAttention(256, encoded, memory_sequence_length=inputs['text_len'])
        decoder_cell = OutputProjectionWrapper(ResidualWrapper(
                MultiRNNCell([GRUCell(384) for _ in range(3)])
        ), 80)

        #def decoder_frame_input(self, inputs, attention):
            #processed_inputs = pre_net(inputs)
            #return tf.concat([processed_inputs, attention], -1)

        decoder_frame_input = \
            lambda inputs, attention: tf.concat([self.pre_net(inputs), attention], -1)

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
            decoder_helper = InferenceHelper(64)
            print(decoder_helper)

        dec = basic_decoder.BasicDecoder(
                cell,
                decoder_helper,
                cell.zero_state(dtype=tf.float32, batch_size=64)
        )

        return dec

    def inference(self, inputs, train=False):
        pre_out = self.pre_net(inputs['text'])

        encoded = CBHG(pre_out, inputs['text_len'])

        dec = self.create_decoder(encoded, train)

        outputs, _ = decoder.dynamic_decode(dec, maximum_iterations=self.config.max_decode_iter)

        return outputs


    def __init__(self, config, inputs):
        self.config = config
        self.output = self.inference(inputs)

class Config(object):
    dropout = 0.5
    max_decode_iter = 400

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
    print(sess.run(model.output))
    





