import logging
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
tf.app.flags.DEFINE_boolean('random_flip_up_down', True,
                           """Whether to random flip up down""")

tf.app.flags.DEFINE_boolean('random_flip_left_right', True,
                           """whether to random flip left and right""")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('image_size',64,
                            """Needs to provide same value as in training.""")
tf.app.flags.DEFINE_boolean('gray', True, "whethet to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 100000, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 10000, "the steps to save")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir','../data/train','the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir','../data/test','the test dataset dir')
tf.app.flags.DEFINE_boolean('restore',False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('val_batch_size', 128, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_string('mode', 'train', 'the run mode')


FLAGS = tf.app.flags.FLAGS



def get_imagesfile(data_dir):
    """
    Return names of training files for `tf.train.string_input_producer`
    """
    filenames = []
    for root, sub_folder, file_list in os.walk(data_dir):
        filenames+=[os.path.join(root, file_path) for file_path in file_list]
    
    labels = [ file_name.split('/')[-2] for file_name in filenames]
    file_labels = [(file, labels[index]) for index, file in enumerate(filenames)]
    random.shuffle(file_labels)

    return file_labels

def pre_process(images):
    if FLAGS.random_flip_up_down:
        images = tf.image.random_flip_up_down(images)
    if FLAGS.random_flip_left_right:
        images = tf.image.random_flip_left_right(images)
    if FLAGS.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.3)
    if FLAGS.random_contrast:
        images = tf.image.random_contrast(images, 0.8, 1.2)
    new_size = tf.constant([FLAGS.image_size,FLAGS.image_size], dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)
    return images



def batch_data(file_labels,sess, batch_size=128):
    image_list = [file_label[0] for file_label in file_labels]
    label_list = [int(file_label[1]) for file_label in file_labels]
    print 'tag2 {0}'.format(len(image_list))
    images_tensor = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(label_list, dtype=tf.int64)
    input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor])

    labels = input_queue[1]
    images_content = tf.read_file(input_queue[0])
    # images = tf.image.decode_png(images_content, channels=1)
    images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
    # images = images / 256
    images =  pre_process(images)
    # print images.get_shape()
    # one hot
    labels = tf.one_hot(labels, 3755)
    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,min_after_dequeue=10000)
    # print 'image_batch', image_batch.get_shape()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch, label_batch, coord, threads

def network(images, labels=None):
    endpoints = {}
    conv_1 = slim.conv2d(images, 32, [3,3],1, padding='SAME')
    max_pool_1 = slim.max_pool2d(conv_1, [2,2],[2,2], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 64, [3,3],padding='SAME')
    max_pool_2 = slim.max_pool2d(conv_2, [2,2],[2,2], padding='SAME')
    flatten = slim.flatten(max_pool_2)
    out = slim.fully_connected(flatten,3755, activation_fn=None)
    global_step = tf.Variable(initial_value=0)
    if labels is not None:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, labels))
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step=global_step)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
    output_score = tf.nn.softmax(out)
    predict_val_top3, predict_index_top3 = tf.nn.top_k(output_score, k=3)

    endpoints['global_step'] = global_step
    if labels is not None:
        endpoints['labels'] = labels
        endpoints['train_op'] = train_op
        endpoints['loss'] = loss
        endpoints['accuracy'] = accuracy
        endpoints['merged_summary_op'] = merged_summary_op
    endpoints['output_score'] = output_score
    endpoints['predict_val_top3'] = predict_val_top3
    endpoints['predict_index_top3'] = predict_index_top3
    return endpoints

def validation():
    # it should be fixed by using placeholder with epoch num in train stage
    sess = tf.Session()
    
    file_labels = get_imagesfile(FLAGS.test_data_dir)
    test_size = len(file_labels)
    print test_size
    val_batch_size = FLAGS.val_batch_size
    test_steps = test_size / val_batch_size
    print test_steps
    # images, labels, coord, threads= batch_data(file_labels, sess)
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    labels = tf.placeholder(dtype=tf.int32, shape=[None,3755])
    # read batch images from file_labels
    # images_batch = np.zeros([128,64,64,1])
    # labels_batch = np.zeros([128,3755])
    # labels_batch[0][20] = 1
    # 
    endpoints = network(images, labels)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        # logger.info("restore from the checkpoint {0}".format(ckpt))
    # logger.info('Start validation')
    final_predict_val = []
    final_predict_index = []
    groundtruth = []
    for i in range(test_steps):
        start = i* val_batch_size
        end = (i+1)*val_batch_size
        images_batch = []
        labels_batch = []
        labels_max_batch = []
        logger.info('=======start validation on {0}/{1} batch========='.format(i, test_steps))
        for j in range(start,end):
            image_path = file_labels[j][0]
            temp_image = Image.open(image_path).convert('L')
            temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size),Image.ANTIALIAS)
            temp_label = np.zeros([3755])
            label = int(file_labels[j][1])
            # print label
            temp_label[label] = 1
            # print "====",np.asarray(temp_image).shape
            labels_batch.append(temp_label)
            # print "====",np.asarray(temp_image).shape
            images_batch.append(np.asarray(temp_image)/255.0)
            labels_max_batch.append(label)
        # print images_batch
        images_batch = np.array(images_batch).reshape([-1, 64, 64, 1])
        labels_batch = np.array(labels_batch)
        batch_predict_val, batch_predict_index = sess.run([endpoints['predict_val_top3'],
                        endpoints['predict_index_top3']], feed_dict={images:images_batch, labels:labels_batch})
        logger.info('=======validation on {0}/{1} batch end========='.format(i, test_steps))
        final_predict_val += batch_predict_val.tolist()
        final_predict_index += batch_predict_index.tolist()
        groundtruth += labels_max_batch
    sess.close()
    return final_predict_val, final_predict_index, groundtruth

    
# def validation(ne):
#     file_labels = get_imagesfile(FLAGS.test_data_dir)
#     val_batch_size = FLAGS.val_batch_size
def inference(image):
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size),Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    sess = tf.Session()
    logger.info('========start inference============')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    endpoints = network(images)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    predict_val, predict_index = sess.run([endpoints['predict_val_top3'],endpoints['predict_index_top3']], feed_dict={images:temp_image})
    sess.close()
    return predict_val, predict_index
    
    


def train():
    sess = tf.Session()
    file_labels = get_imagesfile(FLAGS.train_data_dir)
    images, labels, coord, threads = batch_data(file_labels, sess)
    endpoints = network(images, labels)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    #summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())
    train_writer = tf.train.SummaryWriter('./log' + '/train',sess.graph)
    test_writer = tf.train.SummaryWriter('./log' + '/val')
    start_step = 0
    print 'tag1'
    if FLAGS.restore:
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print "restore from the checkpoint {0}".format(ckpt)
            start_step += int(ckpt.split('-')[-1])
    print 'tag2'
    # logger.info(':::Training Start:::')
    # for i in range(start_step, FLAGS.max_steps):
    try:
        while not coord.should_stop():
        # logger.info('step {0} start'.format(i))
            start_time = time.time()
            _, loss_val, train_summary, step = sess.run([endpoints['train_op'], endpoints['loss'], endpoints['merged_summary_op'], endpoints['global_step']])
            train_writer.add_summary(train_summary, step)
            end_time = time.time()
            logger.info("the step {0} takes {1} loss {2}".format(step, end_time-start_time, loss_val))
            if step > FLAGS.max_steps:
                break
            # logger.info("the step {0} takes {1} loss {2}".format(i, end_time-start_time, loss_val))
            if step % FLAGS.eval_steps == 1:
                accuracy_val,test_summary, step = sess.run([endpoints['accuracy'], endpoints['merged_summary_op'], endpoints['global_step']])
                test_writer.add_summary(test_summary, step)
                logger.info('===============Eval a batch in Train data=======================')
                # print '===============Eval a batch in Train data======================='
                # print 'the step {0} accuracy {1}'.format(step, accuracy_val)
                logger.info( 'the step {0} accuracy {1}'.format(step, accuracy_val))
                logger.info('===============Eval a batch in Train data=======================')
            if step % FLAGS.save_steps == 1:
                logger.info('Save the ckpt of {0}'.format(step))
                # print '===============save=================='
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
    except tf.errors.OutOfRangeError:
        # print "============train finished========="
        logger.info('==================Train Finished================')
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    
def eval_metric(final_predict_index, groundtruth):
    assert len(final_predict_index) == len(groundtruth), 'final_predict_index, size {0} and groundtruth, size {1} must have the same length'.format(len(final_predict_index), len(groundtruth))
    accuracy_cnt = 0
    top3_cnt = 0
    for index, val in enumerate(groundtruth):
        print 'val {0}, final_predict_index {1}'.format(val, final_predict_index[index]) 
        lagrest_predict = final_predict_index[index][0]
        if val == lagrest_predict:
            accuracy_cnt+=1
        if val in final_predict_index[index]:
            top3_cnt += 1
    logger.info('eval on test dataset size: {0}'.format(len(groundtruth)))
    logger.info('The accuracy {0}, the top3 accuracy {1}'.format(accuracy_cnt*1.0/len(groundtruth), top3_cnt*1.0/len(groundtruth)))


def run():
    print FLAGS.mode
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'validation':
        final_predict_val, final_predict_index, groundtruth = validation()
        result = {}
        result['final_predict_val'] = final_predict_val
        result['final_predict_index'] = final_predict_index
        result['groundtruth'] = groundtruth
        result_file = 'result.dict'
        logger.info('Write result into {0}')
        import pickle
        f = open(result_file, 'wb')
        pickle.dump(result, f)
        f.close()
        logger.info('Write file ends')
        eval_metric(final_predict_index, groundtruth)
    elif FLAGS.mode == 'inference':
        print 'inference'
        image_file = '/data/code/dl_opensource/toy_projects/chinese_rec/data/train/01066/325033.png'
        final_predict_val, final_predict_index = inference(image_file)
        logger.info('the result info label {0} predict index {1} predict_val {2}'.format(1066, final_predict_index, final_predict_val))

if __name__ == '__main__':
    run()
