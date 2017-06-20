from __future__ import division
import os
import numpy as np

from utils import heap_add_all
from trainer import Trainer

class MembershipTrainer(Trainer):
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
            train_size=np.inf,
            input_fname_pattern='*.jpg',
            checkpoint_dir="checkpoint"):
        Trainer.__init__(self, sess, name, batch_size, output_height,
                         output_width, z_dim, gf_dim, df_dim, input_dir,
                         train_size, input_fname_pattern, checkpoint_dir)
        self.attack_log = os.path.join(self.checkpoint_dir, 'attack_log.txt')
        self.data = self.train_data + self.test_data
        self.in_train = [1] * len(self.train_data) + [0] * len(self.test_data)
        self.batch_size = batch_size

    def eval(self, epoch, config):
        Trainer.eval(self, epoch, config)
        if np.mod(epoch, 10) == 0:
            guesses = []
            for idx in xrange(len(self.data) // self.batch_size):
                in_score = self.sess.run(self.dcgan.D,
                                         feed_dict={self.dcgan.inputs:
                                                    self.get_images(self.data,
                                                                    idx, config)})
                in_label = self.in_train[idx * self.batch_size:(idx + 1) *
                                         self.batch_size]
                heap_add_all(guesses, zip(in_score, in_label),
                             max_size=len(self.train_data))
            accuracy = sum(label for score, label in guesses) / len(self.train_data)
            print('[*] Attack accuracy: %.2f' % accuracy)
