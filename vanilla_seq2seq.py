from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.rnn import *
from tensorflow.contrib.seq2seq.python.ops \
        import dynamic_attention_wrapper as wrapper, helper, basic_decoder, decoder

import ops
import sys

class Config(object):
    max_decode_iter = 70
    attention_units = 256
    decoder_units = 256
    mel_features = 80
    embed_dim = 256
    fft_size = 1025

    r = 5
    cap_grads = 10
    sampling_prob = 0.5

    lr = 0.0003
    batch_size = 32


class Vanilla_Seq2Seq(object):

    def create_decoder(self, encoded, inputs, train=True):
        config = self.config
        attention_mech = wrapper.BahdanauAttention(
                config.attention_units,
                encoded,
                memory_sequence_length=inputs['text_length']
        )
        decoder_cell = OutputProjectionWrapper(
                InputProjectionWrapper(
                    ResidualWrapper(
                        MultiRNNCell([GRUCell(config.decoder_units) for _ in range(2)])
                ), config.decoder_units)
        , config.fft_size * config.r)

        # feed in rth frame at each time step
        decoder_frame_input = \
            lambda inputs, attention: tf.concat(
                    [tf.slice(inputs, [0, (config.r - 1)*config.fft_size], [-1, -1]), attention]
                , -1)

        cell = wrapper.DynamicAttentionWrapper(
                decoder_cell,
                attention_mech,
                attention_size=config.attention_units,
                cell_input_fn=decoder_frame_input,
                output_attention=False
            )

        # weirdly this worked well with mel features as targets...
        if train:
            decoder_helper = helper.ScheduledOutputTrainingHelper(
                    inputs['stft'],
                    inputs['speech_length'],
                    config.sampling_prob
            )
        else:
            decoder_helper = ops.InferenceHelper(
                    tf.shape(inputs['text'])[0],
                    config.fft_size * config.r
            )

        dec = basic_decoder.BasicDecoder(
                cell,
                decoder_helper,
                cell.zero_state(dtype=tf.float32, batch_size=tf.shape(inputs['text'])[0])
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
            gru_cell = ResidualWrapper(
                    MultiRNNCell([GRUCell(256) for _ in range(2)])
            )
            encoded, _ = tf.nn.dynamic_rnn(
                    gru_cell, 
                    embedded_inputs, 
                    sequence_length=inputs['text_length'],
                    dtype=tf.float32
            )
            print(encoded.shape)

        with tf.variable_scope('decoder'):
            dec = self.create_decoder(encoded, inputs, train)
            (output, _),  _ = decoder.dynamic_decode(dec, maximum_iterations=config.max_decode_iter)
            print(output.shape)

            tf.summary.histogram('output', output)

        return output

    def add_loss_op(self, output, linear):
        loss = tf.reduce_sum(tf.abs(output - linear))
        tf.summary.scalar('loss', loss)
        return loss

    def add_train_op(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            with tf.variable_scope('cap_grads'):
                gvs = [(tf.clip_by_norm(grad, self.config.cap_grads), var) \
                        for grad, var in gvs if grad is not None]

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = opt.apply_gradients(gvs, global_step=self.global_step)
        return train_op

    def __init__(self, config, inputs, train=True):
        self.config = config
        self.output = self.inference(inputs, train)
        if train:
            self.loss = self.add_loss_op(self.output, inputs['stft'])
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
        





