from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


import tensorflow as tf
import time
import json
import sys
sys.path.append('../')
from algorithm import input_data
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("data_dir", "../algorithm/mnist",
                       "mnist data_dir")
tf.flags.DEFINE_string('tensorboard_dir',"./tensorboard", "log dir")
tf.flags.DEFINE_string('checkpoint_dir','./checkpoint', 'checkpoint dir')




def inference(inputs):
    # W = tf.Variable(tf.random_normal([784, 10]))
    # b = tf.Variable(tf.random_normal([10]))
    # y = tf.matmul(inputs, W) + b
    W1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.random_normal([256]))
    W2 = tf.Variable(tf.random_normal([256, 10]))
    b2 = tf.Variable(tf.random_normal([10]))
    lay1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)
    y = tf.add(tf.matmul(lay1, W2),b2)
    return y



def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    input = tf.placeholder(tf.float32, [None, 784])
    y = inference(input)
    y_ = tf.placeholder(tf.float32, [None, 10])
    global_step = tf.Variable(0, name='global_step', trainable=False)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    inference_features = tf.placeholder(tf.float32, [None, 784])
    inference_logits = inference(inference_features)
    inference_softmax = tf.nn.softmax(inference_logits)
    inference_result = tf.argmax(inference_softmax, 1)
    checkpoint_dir = FLAGS.checkpoint_dir
    checkpoint_file = checkpoint_dir + "/checkpoint.ckpt"
    init_op = tf.global_variables_initializer()
    # add the tensors to the tensorboard logs
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    saver = tf.train.Saver()
    keys_placeholder = tf.placeholder("float")
    keys = tf.identity(keys_placeholder)
    tf.add_to_collection("input", json.dumps({'key': keys_placeholder.name, 'features': inference_features.name}))
    tf.add_to_collection('output', json.dumps({'key': keys.name, 'softmax': inference_softmax.name, 'prediction': inference_result.name}))
    with tf.Session() as sess:
        summary_op = tf.summary.merge_all()
        tensorboard_dir = FLAGS.tensorboard_dir
        writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        tf.global_variables_initializer().run()
        for index in range(10000):
            print('process the {}th batch'.format(index))
            start_train = time.time()
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _,summary_val, step  = sess.run([train_step, summary_op, global_step], feed_dict={input: batch_xs, y_: batch_ys})
            writer.add_summary(summary_val, step)
            print('the {0} batch takes time: {1}'.format(index, time.time()-start_train))
        print('the test dataset acc: ', sess.run(accuracy, feed_dict={input: mnist.test.images,y_: mnist.test.labels}))
        saver.save(sess, checkpoint_file)    


if __name__ == '__main__':
  tf.app.run()