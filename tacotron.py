from __future__ import print_function
from __future__ import division

import tensorflow as tf

def highway_layer(inputs, units):
    layer_1 = tf.layers.dense(
            inputs,
            units=units,
            activation=tf.nn.relu
    )

    layer_2 = tf.layers.dense(
            layer_1,
            units=units,
            activation=tf.nn.relu
    )

    out = tf.layers.dense(
            tf.concat([layer_1, layer_2], -1),
            units=units,
            activation=tf.nn.relu
    )
    return  out

def cbhg(inputs, K=16, c=[128,128,128]):
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

    # conv projections
    conv_proj = tf.layers.conv1d(
            conv_bank,
            filters=c[1],
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu
    )
    conv_proj = tf.layers.batch_normalization(conv_proj)

    conv_proj = tf.layers.conv1d(
            conv_bank,
            filters=c[2],
            kernel_size=3,
            padding='same',
    )
    conv_proj = tf.layers.batch_normalization(conv_proj)

    # residual connection
    conv_res = conv_proj + inputs

    print(conv_res.shape)
    return conv_bank

class Tacotron(object):
    def inference():
        pass


# tests
with tf.Session() as sess:
    test = tf.random_normal([64, 129, 1])
    out = cbhg(test)

    tf.global_variables_initializer().run()
    #print(sess.run(out))
    





