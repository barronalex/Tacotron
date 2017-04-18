import tensorflow as tf
import numpy as np
import sys
import os
import data_input

from tqdm import tqdm
import argparse

from tacotron import Tacotron, Config

SAVE_EVERY = 1000
restore = False

def train(config, num_steps=100):

    config.save_path = ''

    filename_queue = tf.train.string_input_producer(['data/squad/proto/train.proto'], num_epochs=None)

    batch_inputs = data_input.batch_inputs(filename_queue)

    ivocab = data_input.load_vocab()
    config.vocab_size = len(ivocab)

    # initialize model
    model = Tacotron(config, batch_inputs, train=True)

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter('log/' + config.save_path + 'train', sess.graph)

        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(max_to_keep=1)

        if restore:
            saver.restore(sess, 'weights/' + config.save_path)

        for step in tqdm(range(num_steps)):
            out = sess.run([model.train_op, model.loss, model.output, model.merged])
            _, loss, output, summary = out
            train_writer.add_summary(summary, step)
            #print(loss)

            if step % SAVE_EVERY == 0 and step != 0:
                if not os.path.exists('weights/' + config.save_path):
                    os.makedirs('weights/' + config.save_path)
                saver.save(sess, 'weights/' + config.save_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = Config()
    train(config)
