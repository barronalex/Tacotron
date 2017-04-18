from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.rnn import *
from tensorflow.contrib.seq2seq.python.ops \
        import dynamic_attention_wrapper as wrapper, helper, basic_decoder, decoder

import ops
import sys

class Config(object):
    max_decode_iter = 400
    attention_units = 256
    decoder_units = 256
    mel_features = 80
    embed_dim = 256

    r = 1
    dropout = 0.5

    lr = 0.001
    batch_size = 32

class Tacotron(object):
    # transformation applied to input character sequence and decoded frame sequence
    def pre_net(self, inputs, units=[256,128], train=True):
        layer_1 = tf.layers.dense(inputs, units[0], activation=tf.nn.relu)
        layer_1 = tf.layers.dropout(layer_1, rate=self.config.dropout, training=train)
        layer_2 = tf.layers.dense(layer_1, units[1], activation=tf.nn.relu)
        layer_2 = tf.layers.dropout(layer_2, rate=self.config.dropout, training=train)
        return layer_2

    def create_decoder(self, encoded, inputs, train=True):
        config = self.config
        attention_mech = wrapper.BahdanauAttention(config.attention_units, encoded, memory_sequence_length=inputs['text_length'])
        decoder_cell = OutputProjectionWrapper(
                InputProjectionWrapper(
                    ResidualWrapper(
                        MultiRNNCell([GRUCell(config.decoder_units) for _ in range(3)])
                ), config.decoder_units)
        , config.mel_features * config.r)

        decoder_frame_input = \
            lambda inputs, attention: tf.concat([self.pre_net(inputs), attention], -1)

        cell = wrapper.DynamicAttentionWrapper(
                decoder_cell,
                attention_mech,
                attention_size=config.attention_units,
                cell_input_fn=decoder_frame_input,
                output_attention=False
        )

        if train:
            decoder_helper = helper.TrainingHelper(inputs['mel'], inputs['speech_length'])
        else:
            decoder_helper = ops.InferenceHelper(config.batch_size)

        dec = basic_decoder.BasicDecoder(
                cell,
                decoder_helper,
                cell.zero_state(dtype=tf.float32, batch_size=config.batch_size)
        )

        return dec

    def inference(self, inputs, train=True):
        config = self.config
        with tf.variable_scope('embedding', initializer=tf.contrib.layers.xavier_initializer()):
            embedding = tf.get_variable('embedding',
                    shape=(config.vocab_size, config.embed_dim), dtype=tf.float32)
            embedded_inputs = tf.nn.embedding_lookup(embedding, inputs['text'])
            print(embedded_inputs.shape)

        with tf.variable_scope('encoder'):
            pre_out = self.pre_net(embedded_inputs)
            encoded = ops.CBHG(pre_out, inputs['text_length'], K=16, c=[128,128,128], gru_units=128)

        with tf.variable_scope('decoder'):
            dec = self.create_decoder(encoded, inputs, train)
            (seq2seq_output, _),  _ = decoder.dynamic_decode(dec, maximum_iterations=config.max_decode_iter)

        with tf.variable_scope('post-process'):
            # TODO rearrange frames so everything makes sense for r > 1
            #decoded = tf.reshape(decoded, [64, -1, 80])
            output = ops.CBHG(seq2seq_output, inputs['speech_length'], K=8, c=[128,256,80])
            output = tf.layers.dense(output, units=1025)

        return seq2seq_output, output

    def add_loss_op(self, seq2seq_output, output, mel, linear):
        seq2seq_loss = tf.reduce_sum(tf.abs(seq2seq_output - mel))
        output_loss = tf.reduce_sum(tf.abs(output - linear))
        loss = seq2seq_loss + output_loss
        tf.summary.scalar('loss', loss)
        return loss

    def add_train_op(self, loss):
        return tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)

    def __init__(self, config, inputs, train=True):
        self.config = config
        self.seq2seq_output, self.output = self.inference(inputs)
        if train:
            self.loss = self.add_loss_op(self.seq2seq_output, self.output, inputs['mel'], inputs['stft'])
            self.train_op = self.add_train_op(self.loss)
        self.merged = tf.summary.merge_all()

if __name__ == '__main__':
    # tests
    with tf.Session() as sess:
        text = tf.ones([32, 129], dtype=tf.int32)
        sl = tf.ones([32], dtype=tf.int32)*50

        mel = tf.random_normal([32, 193, 80])
        stft = tf.random_normal([32, 193, 1025])
        mel_sl = tf.concat(
                [tf.ones([16], dtype=tf.int32)*192, tf.ones([16], dtype=tf.int32)*50],
                axis=0)

        config = Config()
        config.vocab_size = 100
        inputs = {'text': text, 'text_length': sl, 'mel': mel, 'stft': stft, 'speech_length': mel_sl}
        model = Tacotron(config, inputs)

        tf.global_variables_initializer().run()
        loss = sess.run(model.loss)
        print(loss)
        





