import tensorflow as tf
import random
import os
import numpy as np
import tensorflow.contrib.slim as slim
import time
import logging
from PIL import Image


logger = logging.getLogger('Training a chiness write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh = logging.FileHandler('recogniiton.log')
# fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# logger.addHandler(fh)
logger.addHandler(ch)

tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
# tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('image_width',280,
                            "the width of image ")
tf.app.flags.DEFINE_integer('image_height', 28, 'the height of image')
tf.app.flags.DEFINE_boolean('gray', True, "whethet to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 100000, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 10000, "the steps to save")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir','../data/train_data','the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir','../data/test_data','the test dataset dir')
tf.app.flags.DEFINE_boolean('restore',False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('batch_size', 128, 'the batch size during train and val')
tf.app.flags.DEFINE_string('mode', 'train', 'the run mode')

FLAGS = tf.app.flags.FLAGS

class DataIterator:
    def __init__(self, data_dir):
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)
        self.labels = [self.get_label(file_name.split('/')[-1].split('.')[0].split('_')[-1]) for file_name in self.image_names]
        print len(self.labels)
    @property
    def size(self):
        return len(self.labels)

    def get_label(self, str_label):
        """
        Convert the str_label to 10 binary code, 385 to 0001010010
        """
        result = [0]*10
        for i in str_label:
            result[int(i)] = 1
        return result

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_jpeg(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_height, FLAGS.image_width], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        return image_batch, label_batch



def network():
    images = tf.placeholder(dtype=tf.float32, shape=[None, 28, 280, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int32, shape=[None, 10], name='label_batch')
    endpoints = {}
    conv_1 = slim.conv2d(images, 32, [5,5],1, padding='SAME')
    avg_pool_1 = slim.avg_pool2d(conv_1, [2,2],[1,1], padding='SAME')
    conv_2 = slim.conv2d(avg_pool_1, 32, [5,5], 1,padding='SAME')
    avg_pool_2 = slim.avg_pool2d(conv_2, [2,2],[1,1], padding='SAME')
    conv_3 = slim.conv2d(avg_pool_2, 32, [3,3])
    avg_pool_3 = slim.avg_pool2d(conv_3, [2,2], [1,1])
    flatten = slim.flatten(avg_pool_3)
    fc1 = slim.fully_connected(flatten, 512, activation_fn=None)
    out0 = slim.fully_connected(fc1,2, activation_fn=None)
    out1 = slim.fully_connected(fc1,2, activation_fn=None)
    out2 = slim.fully_connected(fc1,2, activation_fn=None)
    out3 = slim.fully_connected(fc1,2, activation_fn=None)
    out4 = slim.fully_connected(fc1,2, activation_fn=None)
    out5 = slim.fully_connected(fc1,2, activation_fn=None)
    out6 = slim.fully_connected(fc1,2, activation_fn=None)
    out7 = slim.fully_connected(fc1,2, activation_fn=None)
    out8 = slim.fully_connected(fc1,2, activation_fn=None)
    out9 = slim.fully_connected(fc1,2, activation_fn=None)
    global_step = tf.Variable(initial_value=0)
    out0_argmax = tf.expand_dims(tf.argmax(out0, 1), 1)
    out1_argmax = tf.expand_dims(tf.argmax(out1, 1), 1)
    out2_argmax = tf.expand_dims(tf.argmax(out2, 1), 1)
    out3_argmax = tf.expand_dims(tf.argmax(out3, 1), 1)
    out4_argmax = tf.expand_dims(tf.argmax(out4, 1), 1)
    out5_argmax = tf.expand_dims(tf.argmax(out5, 1), 1)
    out6_argmax = tf.expand_dims(tf.argmax(out6, 1), 1)
    out7_argmax = tf.expand_dims(tf.argmax(out7, 1), 1)
    out8_argmax = tf.expand_dims(tf.argmax(out8, 1), 1)
    out9_argmax = tf.expand_dims(tf.argmax(out9, 1), 1)
    out_score = tf.concat([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9], axis=1)
    out_final = tf.cast(tf.concat([out0_argmax, out1_argmax, out2_argmax, out3_argmax, out4_argmax, out5_argmax, out6_argmax, out7_argmax, out8_argmax, out9_argmax], axis=1), tf.int32)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out0, labels=tf.one_hot(labels[:,0],depth=2)))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out1, labels=tf.one_hot(labels[:,1],depth=2)))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out2, labels=tf.one_hot(labels[:,2],depth=2)))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out3, labels=tf.one_hot(labels[:,3],depth=2)))
    loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out4, labels=tf.one_hot(labels[:,4],depth=2)))
    loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out5, labels=tf.one_hot(labels[:,5],depth=2)))
    loss6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out6, labels=tf.one_hot(labels[:,6],depth=2)))
    loss7 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out7, labels=tf.one_hot(labels[:,7],depth=2)))
    loss8 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out8, labels=tf.one_hot(labels[:,8],depth=2)))
    loss9 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out9, labels=tf.one_hot(labels[:,9],depth=2)))
    loss_list= [loss, loss1, loss2, loss3,loss4, loss5, loss6, loss7, loss8, loss9]
    loss_sum = tf.reduce_sum(loss_list)
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_sum, global_step=global_step)
    accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(out_final, labels), axis=1), tf.float32))
    tf.summary.scalar('loss_sum', loss_sum)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    endpoints['global_step'] = global_step
    endpoints['images'] = images
    endpoints['labels'] = labels
    endpoints['train_op'] = train_op
    endpoints['loss_sum'] = loss_sum
    endpoints['accuracy'] = accuracy
    endpoints['merged_summary_op'] = merged_summary_op
    endpoints['out_final'] = out_final
    endpoints['out_score'] = out_score
    return endpoints

def validation():
    # it should be fixed by using placeholder with epoch num in train stage
    logger.info("=======Validation Beigin=======")
    test_feeder = DataIterator(data_dir='../data/test_data/')
    predict_labels_list = []
    groundtruth = []
    with tf.Session() as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size,num_epochs=1)
        endpoints = network()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            logger.info('restore from the checkpoint {0}'.format(ckpt))
        logger.info('======Start Validation=======')
        try:
            i = 0
            acc_sum = 0.0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {endpoints['images']:test_images_batch, endpoints['labels']: test_labels_batch}
                labels_batch, predict_labels_batch, acc = sess.run([endpoints['labels'],endpoints['out_final'], endpoints['accuracy']], feed_dict=feed_dict)
                predict_labels_list += predict_labels_batch.tolist()
                groundtruth += labels_batch.tolist()
                acc_sum += acc
                logger.info('the batch {0} takes {1} seconds, accuracy {2}'.format(i, time.time()-start_time, acc))
        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished===================')
            logger.info('The finally accuracy {0}'.format(acc_sum/i))
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'predictions':predict_labels_list, 'gt_labels':groundtruth}

def inference(image):
    logger.info('============inference==========')
    temp_image = Image.open(image).convert('L')
    # temp_image = temp_image.resize((FLAGS.image_height, FLAGS.image_size),Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 28, 280, 1])
    sess = tf.Session()
    logger.info('========start inference============')
    # images = tf.placeholder(dtype=tf.float32, shape=[None, 280, 28, 1])
    endpoints = network()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    feed_dict = {endpoints['images']: temp_image}
    predict_val, predict_index = sess.run([endpoints['out_score'],endpoints['out_final']], feed_dict=feed_dict)
    sess.close()
    return predict_val, predict_index
    
    


def train():
    train_feeder = DataIterator(data_dir=FLAGS.train_data_dir)
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)
    with tf.Session() as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        endpoints = network()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter('./log' + '/train',sess.graph)
        test_writer = tf.summary.FileWriter('./log' + '/val')
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print "restore from the checkpoint {0}".format(ckpt)
                start_step += int(ckpt.split('-')[-1])
        logger.info(':::Training Start:::')
        try:
            while not coord.should_stop():
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {endpoints['images']: train_images_batch, endpoints['labels']: train_labels_batch}
                _, loss_val, train_summary, step = sess.run([endpoints['train_op'], endpoints['loss_sum'], endpoints['merged_summary_op'], endpoints['global_step']], feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                end_time = time.time()
                logger.info("[train] the step {0} takes {1} loss {2}".format(step, end_time-start_time, loss_val))
                if step > FLAGS.max_steps:
                    break
                if step % FLAGS.eval_steps == 1:
                    logger.info('========Begin eval stage =========')
                    start_time = time.time()
                    # can't run 
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    logger.info('[test] gen test batch spend {0}'.format(time.time()-start_time))
                    feed_dict = {
                        endpoints['images']: test_images_batch,
                        endpoints['labels']: test_labels_batch
                    }
                    accuracy_val,test_summary = sess.run([endpoints['accuracy'], endpoints['merged_summary_op']], feed_dict=feed_dict)
                    end_time = time.time()
                    test_writer.add_summary(test_summary, step)
                    logger.info( '[test] the step {0} accuracy {1} spend time {2}'.format(step, accuracy_val, (end_time-start_time)))
                if step % FLAGS.save_steps == 1:
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
        except tf.errors.OutOfRangeError:
            # print "============train finished========="
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
        finally:
            coord.request_stop()
        coord.join(threads)

def run():
    print FLAGS.mode
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'validation':
        result_dict = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}')
        import pickle
        f = open(result_file, 'wb')
        pickle.dump(result_dict, f)
        f.close()
        logger.info('Write file ends')
    elif FLAGS.mode == 'inference':
        print 'inference'
        image_file = '../data/test_data/092e9ae8-ee91-11e6-91c1-525400551618_69128.jpeg'
        final_predict_val, final_predict_index = inference(image_file)
        logger.info('the result info: predict index {0} predict_val {1}'.format( final_predict_index, final_predict_val))

if __name__ == '__main__':
    run()
