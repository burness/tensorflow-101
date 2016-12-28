import tensorflow as tf
from functools import wraps
from flask import Flask, request, jsonify
slim = tf.contrib.slim
from PIL import Image
import sys
sys.path.append("..")
from nets.inception_v3 import *
import numpy as np
import os
import time
"""
Load a tensorflow model and make it available as a REST service
"""
app = Flask(__name__)


class myTfModel(object):
    def __init__(self, model_dir, prefix):
        self.model_dir = model_dir
        self.prefix = prefix
        self.output = {}
        self.load_model()

    def load_model(self):
        sess = tf.Session()
        input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
        arg_scope = inception_v3_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v3(
                input_tensor, is_training=False, num_classes=8)
            saver = tf.train.Saver()
        params_file = tf.train.latest_checkpoint(self.model_dir)
        saver.restore(sess, params_file)
        self.output['sess'] = sess
        self.output['input_tensor'] = input_tensor
        self.output['logits'] = logits
        self.output['end_points'] = end_points
        # return sess, input_tensor, logits, end_points

    def execute(self, data, **kwargs):
        sess = self.output['sess']
        input_tensor = self.output['input_tensor']
        logits = self.output['logits']
        end_points = self.output['end_points']
        # ims = []
        # for i in range(kwargs['batch_size']):
        im = Image.open(data).resize((299, 299))
        im = np.array(im) / 255.0
        im = im.reshape(-1, 299, 299, 3)
        # ims.append(im)
        # ims = np.array(ims)
        # print ims.shape
        start = time.time()
        predict_values, logit_values = sess.run(
            [end_points['Predictions'], logits], feed_dict={input_tensor: im})
        return predict_values
        # print 'the porn score with the {0} is {1} '.format(

    # data, predict_values[1][1])
    # print 'a image take time {0}'.format(time.time() - start)


mymodel = myTfModel('./train_log', 'model.ckpt')


@app.route('/model', methods=['GET', 'POST'])
def apply_model():
    image = request.args.get('image')
    predict_values = mymodel.execute(image, batch_size=1)
    predicted_class = np.argmax(predict_values[0])
    return jsonify(output=int(predicted_class))


if __name__ == '__main__':
    app.run(debug=True)