import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from utils import make_dir, imread, get_image, format_input
from ops import *
from dcgan import DCGAN

class Trainer(object):
    def __init__(
            self,
            sess,
            name,
            batch_size=64,
            output_height=64,
            output_width=64,
            z_dim=100,
            gf_dim=64,
            df_dim=64,
            input_dir=None,
            input_fname_pattern='*',
            checkpoint_dir="checkpoint",
            sample_dir="sample"):
        self.sess = sess
        self.data = [
            fpath for dpath in os.walk(
                os.path.join(
                    "./data",
                    input_dir)) for fpath in glob(
                os.path.join(
                    dpath[0],
                    input_fname_pattern))]
        np.random.shuffle(self.data)

        # check if image is a non-grey image by checking channel number
        img = imread(self.data[0])
        if len(img.shape) >= 3:
            channel = img.shape[-1]
        else:
            channel = 1
        self.grey = (channel == 1)
        self.dcgan = DCGAN(name, batch_size, output_height, output_width,
                           z_dim, gf_dim, df_dim, channel)
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.dcgan.model_id)
        make_dir(self.checkpoint_dir)

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
        self.writer = SummaryWriter("./logs", self.sess.graph)

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

        for epoch in xrange(config.epoch):
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                batch_z, batch_inputs = self.get_batch(
                    idx * self.dcgan.generator.batch_size, config)
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={
                                               self.dcgan.inputs: batch_inputs,
                                               self.dcgan.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network twice to make sure that d_loss does not go
                # to zero
                for i in range(2):
                    _, summary_str = self.sess.run(
                        [g_optim, self.g_sum], feed_dict={self.dcgan.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                errD_fake = self.dcgan.d_loss_fake.eval(
                    {self.dcgan.z: batch_z})
                errD_real = self.dcgan.d_loss_real.eval(
                    {self.dcgan.inputs: batch_inputs})
                errG = self.dcgan.g_loss.eval({self.dcgan.z: batch_z})

                counter += 1
                print(
                    "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
                    (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 0:
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

                if np.mod(counter, 500) == 1:
                    self.save(counter)

    def get_batch(self, start, config):
        return self.get_z(), self.get_inputs(start, config)

    def get_z(self):
        return np.random.uniform(-1, 1, size=(self.dcgan.generator.batch_size,
                                              self.dcgan.generator.z_dim))

    def get_inputs(self, start, config):
        images = [self.read_image(
            f, config) for f in self.data[start:start + self.dcgan.generator.batch_size]]
        return format_input(images, self.grey)

    def read_image(self, fname, config):
        return get_image(
            fname,
            input_height=config.input_height,
            input_width=config.input_width,
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
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
