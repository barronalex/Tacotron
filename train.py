import tensorflow as tf
import numpy as np
import sys
import os
import data_input
import librosa

from tqdm import tqdm
import argparse

import audio

SAVE_EVERY = 2000
RESTORE_FROM = None

def train(model, config, num_steps=1000000):

    sr = 24000 if 'blizzard' in config.data_path else 16000
    meta = data_input.load_meta(config.data_path)
    config.r = meta['r']
    ivocab = meta['vocab']
    config.vocab_size = len(ivocab)

    with tf.Session() as sess:

        inputs, stft_mean, stft_std = data_input.load_from_npy(config.data_path)

        batch_inputs = data_input.build_dataset(sess, inputs)

        # initialize model
        model = model(config, batch_inputs, train=True)

        train_writer = tf.summary.FileWriter('log/' + config.save_path + '/train', sess.graph)

        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=3)

        if config.restore:
            print('restoring weights')
            latest_ckpt = tf.train.latest_checkpoint(
                'weights/' + config.save_path[:config.save_path.rfind('/')]
            )
            if RESTORE_FROM is None:
                if latest_ckpt is not None:
                    saver.restore(sess, latest_ckpt)
            else:
                saver.restore(sess, 'weights/' + config.save_path + '-' + str(RESTORE_FROM))

        lr = model.config.init_lr
        annealing_rate = model.config.annealing_rate
        
        for _ in tqdm(range(num_steps)):
            out = sess.run([
                model.train_op,
                model.global_step,
                model.loss,
                model.output,
                model.alignments,
                model.merged,
                batch_inputs
                ], feed_dict={model.lr: lr})
            _, global_step, loss, output, alignments, summary, inputs = out

            train_writer.add_summary(summary, global_step)

            # detect gradient explosion
            if loss > 1e8 and global_step > 500:
                print('loss exploded')
                break

            if global_step % 1000 == 0:
                lr *= annealing_rate

            if global_step % SAVE_EVERY == 0 and global_step != 0:

                print('saving weights')
                if not os.path.exists('weights/' + config.save_path):
                    os.makedirs('weights/' + config.save_path)
                saver.save(sess, 'weights/' + config.save_path, global_step=global_step)

                print('saving sample')
                # store a sample to listen to
                ideal = audio.invert_spectrogram(inputs['stft'][0]*stft_std + stft_mean)
                sample = audio.invert_spectrogram(output[0]*stft_std + stft_mean)
                attention_plot = data_input.generate_attention_plot(alignments[0])
                step = '_' + str(global_step)
                merged = sess.run(tf.summary.merge(
                    [tf.summary.audio('ideal' + step, ideal[None, :], sr),
                     tf.summary.audio('sample' + step, sample[None, :], sr),
                     tf.summary.image('attention' + step, attention_plot)]
                ))
                train_writer.add_summary(merged, global_step)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-set', default='nancy')
    parser.add_argument('-d', '--debug', type=bool, default=False)
    parser.add_argument('-r', '--restore', type=bool, default=False)
    args = parser.parse_args()

    from models.tacotron import Tacotron, Config
    model = Tacotron
    config = Config()
    config.data_path = 'data/%s/' % args.train_set
    config.restore = args.restore
    if args.debug: 
        config.save_path = 'debug'
    else:
        config.save_path = '%s/tacotron' % args.train_set
    print('Buliding Tacotron')

    train(model, config)
