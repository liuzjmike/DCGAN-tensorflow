import os
from glob import glob
import tensorflow as tf

from ops import *
from utils import make_dir
from generator import Generator
from discriminator import Discriminator


class DCGAN(object):
    def __init__(
            self,
            name,
            batch_size=64,
            output_height=64,
            output_width=64,
            z_dim=100,
            gf_dim=64,
            df_dim=64,
            channel=3):
        """
        Args:
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          channel: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.generator = Generator(batch_size, output_height, output_width,
                                   z_dim, gf_dim, channel)
        self.discriminator = Discriminator(batch_size, output_height,
                                           output_width, df_dim)
        self.model_id = "{}_{}_{}_{}".format(
            name,
            batch_size,
            output_height,
            output_width)

        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(
            tf.float32, [
                self.generator.batch_size, self.generator.output_height, self.generator.output_width, self.generator.channel], name='input')
        self.z = tf.placeholder(
            tf.float32, [None, self.generator.z_dim], name='z')

        with tf.variable_scope("generator") as scope:
            self.G = self.generator.build_graph(self.z, train=True)
            scope.reuse_variables()
            self.sampler = self.generator.build_graph(self.z)

        with tf.variable_scope("discriminator") as scope:
            self.D, self.D_logits = self.discriminator.build_graph(self.inputs)
            scope.reuse_variables()
            self.D_, self.D_logits_ = self.discriminator.build_graph(self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=x, labels=y)
            except BaseException:
                return tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(
                self.D_logits, tf.ones_like(
                    self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(
                self.D_logits_, tf.zeros_like(
                    self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(
                self.D_logits_, tf.ones_like(
                    self.D_)))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
