import tensorflow as tf
import numpy as np
import sys
import os
import data_input
import librosa

from tqdm import tqdm
import argparse

from tacotron import Tacotron, Config

SAVE_EVERY = 500
restore = False
RESTORE_STEP = 99500

def train(config, num_steps=100000):

    config.save_path = ''

    meta = data_input.load_meta()
    assert config.r == meta['r']
    ivocab = meta['vocab']
    config.vocab_size = len(ivocab)

    filename_queue = tf.train.string_input_producer(['data/VCTK-Corpus/train.proto'], num_epochs=None)
    batch_inputs = data_input.batch_inputs(filename_queue, r=config.r)

    # initialize model
    model = Tacotron(config, batch_inputs, train=True)

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter('log/' + config.save_path + 'train', sess.graph)

        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(max_to_keep=3)

        if restore:
            print('restoring weights')
            saver.restore(sess, 'weights/-' + str(RESTORE_STEP))

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
                #sample = output[17]
                #inverted = librosa.istft(sample, win_length=1200, hop_length=300)
                #librosa.output.write_wav('samples/sample_at_{}.wav'.format(step), inverted, 24000)
                #with open('samples/text_at_{}.txt'.format(step), 'w') as sf:
                    #text = [ivocab[w] for w in inputs['text'][17]]
                    #sf.write(str(text))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = Config()
    train(config)
