import os
import time
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from utils import *
from generator import Generator
from discriminator import Discriminator


class DCGAN(object):
    def __init__(
            self,
            dataset,
            batch_size=64,
            input_height=108,
            input_width=108,
            output_height=64,
            output_width=64,
            z_dim=100,
            gf_dim=64,
            df_dim=64,
            c_dim=3,
            checkpoint_dir="checkpoint"):
        """
        Args:
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.generator = Generator(batch_size, output_height, output_width,
                                   z_dim, gf_dim, c_dim)
        self.discriminator = Discriminator(batch_size, output_height,
                                           output_width, df_dim)
        self.model_id = "{}_{}_{}_{}".format(
            self.dataset_name,
            self.batch_size,
            self.output_height,
            self.output_width)
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.model_id)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        build_model()

    def build_model():
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='input')
        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')

        with tf.variable_scope("generator") as scope:
            self.G = self.generator.build_graph(self.z, train=True)
            scope.reuse_variables()
            self.sampler = self.generator.build_graph(self.z)

        with tf.variable_scope("discriminator") as scope:
            self.D, self.D_logits = self.discriminator.build_graph(inputs)
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
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(
                self.D_logits_, tf.ones_like(
                    self.D_)))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(
                self.sess, os.path.join(
                    self.checkpoint_dir, ckpt_name))
            counter = int(
                next(
                    re.finditer(
                        "(\d+)(?!.*\d)",
                        ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
