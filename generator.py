from __future__ import division
import math
import tensorflow as tf

from ops import *

def conv_out_size_same(size, stride):
    return int(math.ceil(size / stride))

class Generator(object):
    def __init__(
            self,
            batch_size=64,
            output_height=64,
            output_width=64,
            z_dim=100,
            filter_dim=64,
            channel=3):
        """
        Args:
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          filter_dim: (optional) Dimension of gen filters in first conv layer. [64]
          channel: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.batch_size = batch_size
        self.output_height = output_height
        self.output_width = output_width
        self.z_dim = z_dim
        self.filter_dim = filter_dim
        self.channel = channel

    def build_graph(self, z, train=False):
        bn0 = batch_norm(name='g_bn0')
        bn1 = batch_norm(name='g_bn1')
        bn2 = batch_norm(name='g_bn2')
        bn3 = batch_norm(name='g_bn3')
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        z_ = linear(z, self.filter_dim * 8 * s_h16 * s_w16, 'g_h0_lin')

        h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.filter_dim * 8])
        h0 = tf.nn.relu(bn0(h0, train=train))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8,
                           self.filter_dim * 4], name='g_h1')
        h1 = tf.nn.relu(bn1(h1, train=train))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4,
                           self.filter_dim * 2], name='g_h2')
        h2 = tf.nn.relu(bn2(h2, train=train))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2,
                           self.filter_dim * 1], name='g_h3')
        h3 = tf.nn.relu(bn3(h3, train=train))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w,
                           self.channel], name='g_h4')

        return tf.nn.tanh(h4)
