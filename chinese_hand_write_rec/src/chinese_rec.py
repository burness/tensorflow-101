# top 1 accuracy 0.5540008378718057 top k accuracy 0.7541684122329284 with default settings.

import tensorflow as tf
import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import pickle
from PIL import Image


logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 8002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 500, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 4000, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', '../data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', '../data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_boolean('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "valid", "test"}')
FLAGS = tf.app.flags.FLAGS


class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)

        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        # print 'image_batch', image_batch.get_shape()
        return image_batch, label_batch


def build_graph(images, labels, keep_prob, top_k, reuse=False):
    # tf.reset_default_graph()
    conv_1 = slim.conv2d(images, 32, [3, 3], 1, padding='SAME', reuse=reuse, scope='conv1')
    max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 64, [3, 3], padding='SAME', reuse=reuse, scope='conv2')
    max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')

    fc1 = slim.flatten(max_pool_2)
    drop1 = slim.dropout(fc1, keep_prob)
    fc2 = slim.fully_connected(drop1, 8000, activation_fn=None, reuse=reuse, scope='fc1')
    drop2 = slim.dropout(fc2, keep_prob)
    logits = slim.fully_connected(drop2, FLAGS.charset_size, activation_fn=None, reuse=reuse, scope='fc2')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)

    probabilities = tf.nn.softmax(logits)
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'images': images,
            'labels': labels,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def train():
    print('Begin training')
    train_feeder = DataIterator(data_dir='../data/train/')
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            images, labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
            graph = build_graph(images, labels, 0.7, 3)

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            saver = tf.train.Saver()

            # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
            start_step = 0
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)
                    print("restore from the checkpoint {0}".format(ckpt))
                    start_step += int(ckpt.split('-')[-1])

            logger.info(':::Training Start:::')
            try:
                i = 0
                while not coord.should_stop():
                    i += 1
                    start_time = time.time()
                    _, loss_val, train_summary, step = sess.run(
                        [graph['train_op'], graph['loss'], graph['merged_summary_op'],
                         graph['global_step']])
                    train_writer.add_summary(train_summary, step)
                    end_time = time.time()
                    logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                    if step > FLAGS.max_steps:
                        break
                    if step % FLAGS.eval_steps == 1:
                        accuracy_train, test_summary, step = sess.run(
                            [graph['accuracy'], graph['merged_summary_op'], graph['global_step']])
                        test_writer.add_summary(test_summary, step)
                        logger.info('===============Eval a batch=======================')
                        logger.info('the step {0} accuracy: train: {1}'
                                    .format(step, accuracy_train))
                        logger.info('===============Eval a batch=======================')
                    if step % FLAGS.save_steps == 0:
                        logger.info('Save the ckpt of {0}'.format(step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                                   global_step=graph['global_step'])
            except tf.errors.OutOfRangeError:
                logger.info('==================Train Finished================')
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])
            finally:
                coord.request_stop()
            coord.join(threads)


def validation():
    print('validation')
    test_feeder = DataIterator(data_dir='../data/test/')

    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            images, labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
            graph = build_graph(images, labels, 1.0, 3)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))

            logger.info(':::Start validation:::')
            try:
                i = 0
                acc_top_1, acc_top_k = 0.0, 0.0
                while not coord.should_stop():
                    i += 1
                    start_time = time.time()
                    batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'],
                                                                           graph['predicted_val_top_k'],
                                                                           graph['predicted_index_top_k'],
                                                                           graph['accuracy'],
                                                                           graph['accuracy_top_k']])
                    final_predict_val += probs.tolist()
                    final_predict_index += indices.tolist()
                    groundtruth += batch_labels.tolist()
                    acc_top_1 += acc_1
                    acc_top_k += acc_k
                    end_time = time.time()
                    logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
                                .format(i, end_time - start_time, acc_1, acc_k))

            except tf.errors.OutOfRangeError:
                logger.info('==================Validation Finished================')
                acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
                acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
                logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
            finally:
                coord.request_stop()
            coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}


def inference(image):
    print('inference')
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            logger.info('========start inference============')
            images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
            # Pass a shadow label 0. This label will not affect the computation graph.
            graph = build_graph(images, [0], 1.0, 3)
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
            predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                                  feed_dict={images: temp_image})
    return predict_val, predict_index


def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'validation':
        dct = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(dct, f)
        logger.info('Write file ends')
    elif FLAGS.mode == 'inference':
        image_path = '../data/test/00190/13320.png'
        final_predict_val, final_predict_index = inference(image_path)
        logger.info('the result info label {0} predict index {1} predict_val {2}'.format(190, final_predict_index,
                                                                                         final_predict_val))

if __name__ == "__main__":
    tf.app.run()
