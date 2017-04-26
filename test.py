import tensorflow as tf
import numpy as np
import sys
import os
import data_input
import librosa

from tqdm import tqdm
import argparse

import audio

RESTORE_STEP = 500

def test(model, config, prompt_file, num_steps=100000):

    meta = data_input.load_meta()
    assert config.r == meta['r']
    ivocab = meta['vocab']
    config.vocab_size = len(ivocab)

    batch_inputs = data_input.load_prompts(prompt_file, ivocab)

    # initialize model
    model = model(config, batch_inputs, train=False)

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter('log/' + config.save_path + '/test', sess.graph)

        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()

        print('restoring weights')
        saver.restore(sess, 'weights/' + config.save_path + '-' + str(RESTORE_STEP))

        try:
            while(True):
                out = sess.run([
                    model.global_step,
                    model.output,
                    model.merged,
                    batch_inputs
                ])
                _, global_step, output, summary, inputs = out

                print('saving samples')
                for out, text in zip(outputs, inputs['text']):
                    # store a sample to listen to
                    sample = audio.invert_spectrogram(out)
                    merged = sess.run(tf.summary.merge(
                         tf.summary.audio(text, sample[None, :], 16000)
                    ))
                    train_writer.add_summary(merged, 0)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prompts')
    parser.add_argument('-m', '--model', default='tacotron')
    args = parser.parse_args()

    if args.model == 'tacotron':
        from tacotron import Tacotron, Config
        model = Tacotron
        config = Config()
        config.save_path = 'tacotron'
        print('Buliding Tacotron')
    else:
        from vanilla_seq2seq import Vanilla_Seq2Seq, Config
        model = Vanilla_Seq2Seq
        config = Config()
        config.save_path = 'vanilla_seq2seq/scheduled_sample'
        print('Buliding Vanilla_Seq2Seq')

    test(model, config, args.prompts)
