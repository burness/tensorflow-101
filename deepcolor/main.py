import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import math
from random import randint

import ops
from ops import *
from utils import *

def imageblur(cimg, sampling=False):
    if sampling:
        cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
    else:
        for i in xrange(30):
            randx = randint(0,205)
            randy = randint(0,205)
            cimg[randx:randx+50, randy:randy+50] = 255
    return cv2.blur(cimg,(100,100))

class Color():
    def __init__(self, imgsize=256, batchsize=4):
        self.batch_size = batchsize
        self.batch_size_sqrt = int(math.sqrt(self.batch_size))
        self.image_size = imgsize
        self.output_size = imgsize

        self.gf_dim = 64
        self.df_dim = 64

        self.input_colors = 1
        self.input_colors2 = 3
        self.output_colors = 3

        self.l1_scaling = 100

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors])
        self.color_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors2])
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_colors])

        combined_preimage = tf.concat(axis=3, values=[self.line_images, self.color_images])
        # combined_preimage = tf.concat([self.line_images, self.color_images], 3)
        # combined_preimage = self.line_images

        self.generated_images = self.generator(combined_preimage)

        self.real_AB = tf.concat(axis=3, values=[combined_preimage, self.real_images])
        self.fake_AB = tf.concat(axis=3, values=[combined_preimage, self.generated_images])

        # self.real_AB = tf.concat([combined_preimage, self.real_images], 3)
        # self.fake_AB = tf.concat([combined_preimage, self.generated_images], 3)

        self.disc_true, disc_true_logits = self.discriminator(self.real_AB, reuse=False)
        self.disc_fake, disc_fake_logits = self.discriminator(self.fake_AB, reuse=True)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_true_logits, labels=tf.ones_like(disc_true_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits))) \
                        + self.l1_scaling * tf.reduce_mean(tf.abs(self.real_images - self.generated_images))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)


    def discriminator(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv')) # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'))) # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'))) # h2 is (32 x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))) # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            return tf.nn.sigmoid(h4), h4

    def generator(self, img_in):
        with tf.variable_scope("generator") as scope:
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv') # e1 is (128 x 128 x self.gf_dim)
            e2 = bn(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')) # e2 is (64 x 64 x self.gf_dim*2)
            e3 = bn(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')) # e3 is (32 x 32 x self.gf_dim*4)
            e4 = bn(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')) # e4 is (16 x 16 x self.gf_dim*8)
            e5 = bn(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')) # e5 is (8 x 8 x self.gf_dim*8)


            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e5), [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = bn(self.d4)
            d4 = tf.concat(axis=3, values=[d4, e4])
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = bn(self.d5)
            d5 = tf.concat(axis=3, values=[d5, e3])
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = bn(self.d6)
            d6 = tf.concat(axis=3, values=[d6, e2])
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = bn(self.d7)
            d7 = tf.concat(axis=3, values=[d7, e1])
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, s, s, self.output_colors], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(self.d8)


    def imageblur(self, cimg, sampling=False):
        if sampling:
            cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
        else:
            for i in xrange(30):
                randx = randint(0,205)
                randy = randint(0,205)
                cimg[randx:randx+50, randy:randy+50] = 255
        return cv2.blur(cimg,(100,100))

    def train(self):
        self.loadmodel()

        data = glob(os.path.join("imgs", "*.jpg"))
        print data[0]
        base = np.array([get_image(sample_file) for sample_file in data[0:self.batch_size]])
        base_normalized = base/255.0

        base_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in base]) / 255.0
        base_edge = np.expand_dims(base_edge, 3)

        base_colors = np.array([self.imageblur(ba) for ba in base]) / 255.0

        ims("results/base.png",merge_color(base_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims("results/base_line.jpg",merge(base_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
        ims("results/base_colors.jpg",merge_color(base_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))

        datalen = len(data)

        for e in xrange(20000):
            for i in range(datalen / self.batch_size):
                batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = np.array([get_image(batch_file) for batch_file in batch_files])
                batch_normalized = batch/255.0

                batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in batch]) / 255.0
                batch_edge = np.expand_dims(batch_edge, 3)

                batch_colors = np.array([self.imageblur(ba) for ba in batch]) / 255.0

                d_loss, _ = self.sess.run([self.d_loss, self.d_optim], feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge, self.color_images: batch_colors})
                g_loss, _ = self.sess.run([self.g_loss, self.g_optim], feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge, self.color_images: batch_colors})

                print "%d: [%d / %d] d_loss %f, g_loss %f" % (e, i, (datalen/self.batch_size), d_loss, g_loss)

                if i % 100 == 0:
                    recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: base_normalized, self.line_images: base_edge, self.color_images: base_colors})
                    ims("results/"+str(e*100000 + i)+".jpg",merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))

                if i % 500 == 0:
                    self.save("./checkpoint", e*100000 + i)

    def loadmodel(self, load_discrim=True):
        self.sess = tf.Session()
        try:
            self.sess.run(tf.global_variables_initializer())
        except AttributeError:
            self.sess.run(tf.initialize_all_variables())


        if load_discrim:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(self.g_vars)

        if self.load("./checkpoint"):
            print "Loaded"
        else:
            print "Load failed"

    def sample(self):
        self.loadmodel()

        data = glob(os.path.join("imgs", "*.jpg"))

        datalen = len(data)

        for i in range(min(100,datalen / self.batch_size)):
            batch_files = data[i*self.batch_size:(i+1)*self.batch_size]
            batch = np.array([get_image(batch_file) for batch_file in batch_files])
            batch_normalized = batch/255.0

            batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in batch]) / 255.0
            batch_edge = np.expand_dims(batch_edge, 3)

            batch_colors = np.array([self.imageblur(ba,True) for ba in batch]) / 255.0

            recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: batch_normalized, self.line_images: batch_edge, self.color_images: batch_colors})
            ims("results/sample_"+str(i)+".jpg",merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("results/sample_"+str(i)+"_origin.jpg",merge_color(batch_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("results/sample_"+str(i)+"_line.jpg",merge_color(batch_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))
            ims("results/sample_"+str(i)+"_color.jpg",merge_color(batch_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))


    def save(self, checkpoint_dir, step):
        model_name = "model"
        model_dir = "tr"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "tr"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python main.py [train, sample]"
    else:
        cmd = sys.argv[1]
        if cmd == "train":
            c = Color()
            c.train()
        elif cmd == "sample":
            c = Color()
            c.sample()
        else:
            print "Usage: python main.py [train, sample]"
