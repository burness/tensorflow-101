from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("data_dir", "./mnist",
                       "mnist data_dir")


def main(_):
  mnist = read_data_sets(FLAGS.data_dir, one_hot=True)
  x = tf.placeholder(tf.float32, [None, 784])
  W1 = tf.Variable(tf.random_normal([784, 256]))
  b1 = tf.Variable(tf.random_normal([256]))
  W2 = tf.Variable(tf.random_normal([256, 10]))
  b2 = tf.Variable(tf.random_normal([10]))
  lay1 = tf.nn.relu(tf.matmul(x, W1) + b1)
  y = tf.add(tf.matmul(lay1, W2),b2)

  y_ = tf.placeholder(tf.float32, [None, 10])
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  for index in range(10000): 
    # print('process the {}th batch'.format(index))
    start_train = time.time()
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # print('the {0} batch takes time: {1}'.format(index, time.time()-start_train))

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  tf.app.run()