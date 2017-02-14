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
tf.app.flags.DEFINE_boolean('val_batch_size', 128, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_string('mode', 'train', 'the run mode')


FLAGS = tf.app.flags.FLAGS

def get_label(str_label):
    """
    Convert the str_label to 10 binary code, 385 to 0001010010
    """
    result = [0]*10
    for i in str_label:
        result[int(i)] = 1
    return result


def get_imagesfile(data_dir):
    """
    Return names of training files for `tf.train.string_input_producer`
    """
    filenames = []
    for root, sub_folder, file_list in os.walk(data_dir):
        filenames+=[os.path.join(root, file_path) for file_path in file_list]
    
    labels = [ file_name.split('/')[-1].split('.')[0].split('_')[-1] for file_name in filenames]
    binary_labels = []
    for str_label in labels:
        binary_labels.append(get_label(str_label))
    file_labels = zip(filenames, binary_labels)
    random.shuffle(file_labels)
    print 'file_labels len {0}, file_labels sample {1}'.format(len(file_labels), file_labels[0])
    return file_labels

def pre_process(images):
    if FLAGS.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.3)
    new_size = tf.constant([FLAGS.image_height,FLAGS.image_width], dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)
    return images



def batch_data(file_labels,sess, batch_size=128):
    image_list = [file_label[0] for file_label in file_labels]
    label_list = [file_label[1] for file_label in file_labels]
    print 'tag2 {0}'.format(len(image_list))
    images_tensor = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(label_list, dtype=tf.int32)
    input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor])

    labels = input_queue[1]
    images_content = tf.read_file(input_queue[0])
    images = tf.image.convert_image_dtype(tf.image.decode_jpeg(images_content, channels=1), tf.float32)
    images =  pre_process(images)
    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,min_after_dequeue=10000)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch, label_batch, coord, threads

# def compute_accuracy(out_list, labels):

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
        return image_batch, label_batch



def network(images, labels=None):
    endpoints = {}
    conv_1 = slim.conv2d(images, 32, [5,5],1, padding='SAME')
    avg_pool_1 = slim.avg_pool2d(conv_1, [2,2],[1,1], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 32, [5,5], 1,padding='SAME')
    avg_pool_2 = slim.avg_pool2d(conv_2, [2,2],[1,1], padding='SAME')
    conv_3 = slim.conv2d(max_pool_2, 32, [3,3])
    avg_pool_3 = slim.avg_pool2d(conv_3, [2,2], [1,1])
    flatten = slim.flatten(max_pool_2)
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
    out_score = tf.concat(1, [out0, out1, out2, out3, out4, out5, out6, out7, out8, out9])
    out_final = tf.cast(tf.concat(1, [out0_argmax, out1_argmax, out2_argmax, out3_argmax, out4_argmax, out5_argmax, out6_argmax, out7_argmax, out8_argmax, out9_argmax]), tf.int32)

    print 'out0 shape {0}, labels[:, 0, ]  shape {1}'.format(out0.get_shape(), labels.get_shape())    
    if labels is not None:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out0, tf.one_hot(labels[:,0],depth=2)))
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out1, tf.one_hot(labels[:,1],depth=2)))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out2, tf.one_hot(labels[:,2],depth=2)))
        loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out3, tf.one_hot(labels[:,3],depth=2)))
        loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out4, tf.one_hot(labels[:,4],depth=2)))
        loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out5, tf.one_hot(labels[:,5],depth=2)))
        loss6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out6, tf.one_hot(labels[:,6],depth=2)))
        loss7 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out7, tf.one_hot(labels[:,7],depth=2)))
        loss8 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out8, tf.one_hot(labels[:,8],depth=2)))
        loss9 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out9, tf.one_hot(labels[:,9],depth=2)))
        loss_list= [loss, loss1, loss2, loss3,loss4, loss5, loss6, loss7, loss8, loss9]
        loss_sum = tf.reduce_sum(loss_list)
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_sum, global_step=global_step)
        accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(out_final, labels), axis=1), tf.float32))
        tf.summary.scalar('loss_sum', loss_sum)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()

    endpoints['global_step'] = global_step
    if labels is not None:
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
    sess = tf.Session()
    image_width = FLAGS.image_width
    image_height = FLAGS.image_height
    
    file_labels = get_imagesfile(FLAGS.test_data_dir)
    test_size = len(file_labels)
    print test_size
    val_batch_size = FLAGS.val_batch_size
    test_steps = test_size / val_batch_size
    print test_steps
    images = tf.placeholder(dtype=tf.float32, shape=[None, image_width, image_height, 1])
    labels = tf.placeholder(dtype=tf.int32, shape=[None,10])
    endpoints = network(images, labels)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    final_predict_val = []
    final_predict_index = []
    groundtruth = []
    for i in range(test_steps):
        start = i* val_batch_size
        end = (i+1)*val_batch_size
        images_batch = []
        labels_batch = []
        logger.info('=======start validation on {0}/{1} batch========='.format(i, test_steps))
        for j in range(start,end):
            image_path = file_labels[j][0]
            temp_image = Image.open(image_path).convert('L')
            temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size),Image.ANTIALIAS)
            label = file_labels[j][1]
            images_batch.append(np.asarray(temp_image)/255.0)
            labels_batch.append(label)
        images_batch = np.array(images_batch).reshape([-1, image_width, image_height, 1])
        labels_batch = np.array(labels_batch)
        batch_predict_val, batch_predict_index = sess.run([endpoints['out_score'],
                        endpoints['out_final']], feed_dict={images:images_batch, labels:labels_batch})
        logger.info('=======validation on {0}/{1} batch end========='.format(i, test_steps))
        final_predict_val += batch_predict_val.tolist()
        final_predict_index += batch_predict_index.tolist()
        groundtruth += labels_batch
    sess.close()
    return final_predict_val, final_predict_index, groundtruth

    

def inference(image):
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size),Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 280, 28, 1])
    sess = tf.Session()
    logger.info('========start inference============')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 280, 28, 1])
    endpoints = network(images)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    predict_val, predict_index = sess.run([endpoints['out_score'],endpoints['out_final']], feed_dict={images:temp_image})
    sess.close()
    return predict_val, predict_index
    
    


def train():
    sess = tf.Session()
    logger.info()
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
            _, loss_val, train_summary, step = sess.run([endpoints['train_op'], endpoints['loss_sum'], endpoints['merged_summary_op'], endpoints['global_step']])
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
    
# def eval_metric(final_predict_index, groundtruth):
#     assert len(final_predict_index) == len(groundtruth), 'final_predict_index, size {0} and groundtruth, size {1} must have the same length'.format(len(final_predict_index), len(groundtruth))
#     accuracy_cnt = 0
#     top3_cnt = 0
#     for index, val in enumerate(groundtruth):
#         print 'val {0}, final_predict_index {1}'.format(val, final_predict_index[index]) 
#         lagrest_predict = final_predict_index[index][0]
#         if val == lagrest_predict:
#             accuracy_cnt+=1
#         if val in final_predict_index[index]:
#             top3_cnt += 1
#     logger.info('eval on test dataset size: {0}'.format(len(groundtruth)))
#     logger.info('The accuracy {0}, the top3 accuracy {1}'.format(accuracy_cnt*1.0/len(groundtruth), top3_cnt*1.0/len(groundtruth)))


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
