import tensorflow as tf
import numpy as np
import sys
import os
import data_input
import librosa

from tqdm import tqdm
import argparse

import audio

SAVE_EVERY = 500
restore = False

def train(model, config, num_steps=100000):

    meta = data_input.load_meta()
    assert config.r == meta['r']
    ivocab = meta['vocab']
    config.vocab_size = len(ivocab)


    with tf.Session() as sess:

        inputs = data_input.load_from_npy('data/blizzard/')
        queue, batch_inputs = data_input.build_queue(sess, inputs)

        # initialize model
        model = model(config, batch_inputs, train=True)

        train_writer = tf.summary.FileWriter('log/' + config.save_path + '/train', sess.graph)

        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(max_to_keep=3)

        if restore:
            print('restoring weights')
            latest_ckpt = tf.train.latest_checkpoint(
                'weights/' + config.save_path[:config.save_path.rfind('/')]
            )
            saver.restore(sess, latest_ckpt)

        for _ in tqdm(range(num_steps)):
            out = sess.run([
                model.train_op,
                model.global_step,
                model.loss,
                model.output,
                model.merged,
                batch_inputs
            ])
            _, global_step, loss, output, summary, inputs = out
            train_writer.add_summary(summary, global_step)

            if global_step % SAVE_EVERY == 0 and global_step != 0:
                print('saving weights')
                if not os.path.exists('weights/' + config.save_path):
                    os.makedirs('weights/' + config.save_path)
                saver.save(sess, 'weights/' + config.save_path, global_step=global_step)
                print('saving sample')
                # store a sample to listen to
                sample = audio.invert_spectrogram(output[17])
                ideal = audio.invert_spectrogram(inputs['stft'][17])
                merged = sess.run(tf.summary.merge(
                    [tf.summary.audio('ideal{}'.format(global_step), ideal[None, :], 16000),
                     tf.summary.audio('sample{}'.format(global_step), sample[None, :], 16000)]
                ))
                train_writer.add_summary(merged, global_step)

        sess.run(queue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        config.save_path = 'blizzard/vanilla_seq2seq'
        print('Buliding Vanilla_Seq2Seq')

    train(model, config)
