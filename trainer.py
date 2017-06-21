import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from utils import make_dir, get_files, imread, get_image, format_input
from meta_saver import save_meta
from ops import *
from dcgan import DCGAN

def get_data(base_dir, pattern):
    return get_files(os.path.join('./data', base_dir), pattern)

class Trainer(object):
    def __init__(
            self,
            sess,
            name,
            train_set,
            test_set,
            batch_size=64,
            output_height=64,
            output_width=64,
            z_dim=100,
            gf_dim=64,
            df_dim=64,
            train_size=None,
            input_fname_pattern='*.jpg',
            checkpoint_dir="checkpoint"):
        self.sess = sess
        self.train_data = get_data(train_set, input_fname_pattern)
        np.random.shuffle(self.train_data)
        if train_size != None:
            self.train_data = self.train_data[:train_size]

        # check if image is a non-grey image by checking channel number
        img = imread(self.train_data[0])
        if len(img.shape) >= 3:
            channel = img.shape[-1]
        else:
            channel = 1
        self.input_width = img.shape[0]
        self.input_height = img.shape[1]
        self.grey = (channel == 1)
        self.dcgan = DCGAN(name, batch_size, output_height, output_width,
                           z_dim, gf_dim, df_dim, channel)
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.dcgan.model_id)
        make_dir(self.checkpoint_dir)
        save_meta(self.dcgan, self.checkpoint_dir)

        self.saver = tf.train.Saver()

        self.init_summary()

    def init_summary(self):
        self.z_sum = histogram_summary("z", self.dcgan.z)
        self.d_sum = histogram_summary("d", self.dcgan.D)
        self.d__sum = histogram_summary("d_", self.dcgan.D_)
        self.G_sum = image_summary("G", self.dcgan.G)
        self.d_loss_real_sum = scalar_summary(
            "d_loss_real", self.dcgan.d_loss_real)
        self.d_loss_fake_sum = scalar_summary(
            "d_loss_fake", self.dcgan.d_loss_fake)
        self.d_loss_sum = scalar_summary("d_loss", self.dcgan.d_loss)
        self.g_loss_sum = scalar_summary("g_loss", self.dcgan.g_loss)
        self.g_sum = merge_summary(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        # self.writer = SummaryWriter("./logs", self.sess.graph)

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(
            config.learning_rate,
            beta1=config.beta1) .minimize(
            self.dcgan.d_loss,
            var_list=self.dcgan.d_vars)
        g_optim = tf.train.AdamOptimizer(
            config.learning_rate,
            beta1=config.beta1) .minimize(
            self.dcgan.g_loss,
            var_list=self.dcgan.g_vars)
        try:
            tf.global_variables_initializer().run()
        except BaseException:
            tf.initialize_all_variables().run()

        sample_z, sample_inputs = self.get_batch(0, config)

        start_time = time.time()
        load_success, counter = self.load()
        if load_success:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        sample_dir = os.path.join(config.sample_dir, self.dcgan.model_id)
        make_dir(sample_dir)

        batch_idxs = len(self.train_data) // self.dcgan.generator.batch_size

        for epoch in xrange(counter, config.epoch):
            np.random.shuffle(self.train_data)
            for idx in xrange(batch_idxs):
                batch_z, batch_inputs = self.get_batch(idx, config)
                # Update D network
                self.sess.run(d_optim, feed_dict={self.dcgan.inputs:
                                                  batch_inputs, self.dcgan.z: batch_z})

                # Update G network twice to make sure that d_loss does not go
                # to zero
                for i in range(2):
                    self.sess.run(g_optim, feed_dict={self.dcgan.z: batch_z})

                errD_fake = self.dcgan.d_loss_fake.eval(
                    {self.dcgan.z: batch_z})
                errD_real = self.dcgan.d_loss_real.eval(
                    {self.dcgan.inputs: batch_inputs})
                errG = self.dcgan.g_loss.eval({self.dcgan.z: batch_z})

                print(
                    "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
                    (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

            try:
                samples, d_loss, g_loss = self.sess.run(
                    [self.dcgan.sampler, self.dcgan.d_loss, self.dcgan.g_loss],
                    feed_dict={
                        self.dcgan.z: sample_z,
                        self.dcgan.inputs: sample_inputs,
                    },
                )
                manifold_h = int(
                    np.ceil(np.sqrt(samples.shape[0])))
                manifold_w = int(
                    np.floor(np.sqrt(samples.shape[0])))
                save_images(samples,
                            [manifold_h,
                             manifold_w],
                            './{}/train_{:02d}_{:04d}.png'.format(sample_dir,
                                                                  epoch,
                                                                  idx))
                print(
                    "[Sample] d_loss: %.8f, g_loss: %.8f" %
                    (d_loss, g_loss))
            except BaseException:
                print("one pic error!...")

            self.eval(epoch, config)

    def eval(self, epoch, config):
        if np.mod(epoch, 10) == 9:
            self.save(epoch)

    def get_batch(self, batch, config):
        return self.get_z(), self.get_inputs(batch, config)

    def get_z(self):
        return np.random.uniform(-1, 1, size=(self.dcgan.generator.batch_size,
                                              self.dcgan.generator.z_dim))

    def get_inputs(self, batch, config):
        return self.get_images(self.train_data, batch, config)

    def get_images(self, data, batch, config):
        batch_size = self.dcgan.generator.batch_size
        images = [self.read_image(
            f, config) for f in data[batch * batch_size:(batch + 1) * batch_size]]
        return format_input(images, self.grey)

    def read_image(self, fname, config):
        return get_image(
            fname,
            input_height=self.input_height,
            input_width=self.input_width,
            resize_height=self.dcgan.generator.output_height,
            resize_width=self.dcgan.generator.output_width,
            crop=config.crop,
            grayscale=self.grey)

    def save(self, step):
        model_name = "DCGAN.model"
        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def load(self):
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
            return True, counter + 1
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
