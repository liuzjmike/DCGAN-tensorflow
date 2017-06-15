import tensorflow as tf

from ops import *

class Discriminator(object):

    def __init__(
            self,
            batch_size=64,
            input_height=64,
            input_width=64,
            filter_dim=64):
        self.batch_size=batch_size
        self.input_height=input_height
        self.input_width=input_width
        self.filter_dim=filter_dim

    def build_graph(self, image):
        bn1 = batch_norm(name='d_bn1')
        bn2 = batch_norm(name='d_bn2')
        bn3 = batch_norm(name='d_bn3')
        h0 = lrelu(conv2d(image, self.filter_dim, name='d_h0_conv'))
        h1 = lrelu(bn1(conv2d(h0, self.filter_dim * 2, name='d_h1_conv')))
        h2 = lrelu(bn2(conv2d(h1, self.filter_dim * 4, name='d_h2_conv')))
        h3 = lrelu(bn3(conv2d(h2, self.filter_dim * 8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
        return tf.nn.sigmoid(h4), h4
