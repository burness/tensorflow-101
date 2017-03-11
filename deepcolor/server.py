from bottle import route, run, template, static_file, get, post, request, BaseRequest
import urllib2
import cv2
import numpy as np
import re
import base64

import main
from main import *

BaseRequest.MEMFILE_MAX = 1000 * 1000

c = Color(512, 1)
c.loadmodel(False)

@route('/<filename:path>')
def send_static(filename):
    return static_file(filename, root='web/')

@route('/draw')
def send_static():
    return static_file("draw.html", root='web/')

@route('/')
def send_static():
    return static_file("index.html", root='web/')

def imageblur(cimg, sampling=False):
    if sampling:
        cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
    else:
        for i in xrange(30):
            randx = randint(0,205)
            randy = randint(0,205)
            cimg[randx:randx+50, randy:randy+50] = 255
    return cv2.blur(cimg,(100,100))

@route("/standard_sanae", method="POST")
def do_uploadtl():
    lines_img = cv2.imread("web/image_examples/sanae.png", 1)
    lines_img = np.array(cv2.resize(lines_img, (512,512)))
    lines_img = cv2.adaptiveThreshold(cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    lines_img = cv2.merge((lines_img,lines_img,lines_img,255 - lines_img))
    cnt = cv2.imencode(".png",lines_img)[1]
    return base64.b64encode(cnt)

@route("/standard_armscross", method="POST")
def do_uploadtl():
    lines_img = cv2.imread("web/image_examples/armscross.png", 1)
    lines_img = np.array(cv2.resize(lines_img, (512,512)))
    lines_img = cv2.adaptiveThreshold(cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    lines_img = cv2.merge((lines_img,lines_img,lines_img,255 - lines_img))
    cnt = cv2.imencode(".png",lines_img)[1]
    return base64.b64encode(cnt)

@route("/standard_picasso", method="POST")
def do_uploadtl():
    lines_img = cv2.imread("web/image_examples/picasso.png", 1)
    lines_img = np.array(cv2.resize(lines_img, (512,512)))
    lines_img = cv2.adaptiveThreshold(cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    lines_img = cv2.merge((lines_img,lines_img,lines_img,255 - lines_img))
    cnt = cv2.imencode(".png",lines_img)[1]
    return base64.b64encode(cnt)

@route("/upload_toline", method="POST")
def do_uploadtl():
    print "Parsing line"
    img = request.files.get('img')
    lines_img = cv2.imdecode(np.fromstring(img.file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    lines_img = np.array(cv2.resize(lines_img, (512,512)))
    lines_img = cv2.adaptiveThreshold(cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    lines_img = cv2.merge((lines_img,lines_img,lines_img,255 - lines_img))
    cnt = cv2.imencode(".png",lines_img)[1]
    return base64.b64encode(cnt)

def imageblur(cimg, sampling=False):
    if sampling:
        cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
    else:
        for i in xrange(30):
            randx = randint(0,205)
            randy = randint(0,205)
            cimg[randx:randx+50, randy:randy+50] = 255
    return cv2.blur(cimg,(100,100))

@route('/upload_canvas', method='POST')
def do_uploadc():
    print "Got it"
    # lines = request.files.get('lines')
    # colors = request.files.get('colors')
    line_data = request.forms.get("lines")
    line_data = re.sub('^data:image/.+;base64,', '', line_data)
    line_s = base64.b64decode(line_data)
    line_img = np.fromstring(line_s, dtype=np.uint8)
    line_img = cv2.imdecode(line_img, -1)

    color_data = request.forms.get("colors")
    color_data = re.sub('^data:image/.+;base64,', '', color_data)
    color_s = base64.b64decode(color_data)
    color_img = np.fromstring(color_s, dtype=np.uint8)
    color_img = cv2.imdecode(color_img, -1)

    # for c in range(0,3):
    color_img = color_img * (line_img[:,:] / 255.0)

    lines_img = np.array(cv2.resize(line_img, (512,512)))
    # lines_img = cv2.adaptiveThreshold(lines_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    lines_img = np.array([lines_img]) / 255.0
    lines_img = lines_img[:,:,:,0]
    lines_img = np.expand_dims(lines_img, 3)

    colors_img = np.array(cv2.resize(color_img, (512,512)))
    # colors_img = cv2.blur(colors_img, (100, 100))
    colors_img = imageblur(colors_img, True)
    colors_img = np.array([colors_img]) / 255.0
    colors_img = colors_img[:,:,:,0:3]
    generated = c.sess.run(c.generated_images, feed_dict={c.line_images: lines_img, c.color_images: colors_img})
    cnt = cv2.imencode(".png",generated[0]*255)[1]
    return base64.b64encode(cnt)

@route('/upload_origin', method='POST')
def do_uploado():
    lines = request.files.get('lines')
    colors = request.files.get('colors')

    lines_img = cv2.imdecode(np.fromstring(lines.file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    lines_img = np.array(cv2.resize(lines_img, (512,512)))
    lines_img = cv2.adaptiveThreshold(cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    lines_img = np.array([lines_img]) / 255.0
    lines_img = np.expand_dims(lines_img, 3)

    colors_img = cv2.imdecode(np.fromstring(colors.file.read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    colors_img = np.array(cv2.resize(colors_img, (512,512)))
    colors_img = cv2.blur(colors_img, (100, 100))
    colors_img = np.array([colors_img]) / 255.0

    cv2.imwrite("uploaded/lines.jpg", lines_img[0]*255)
    cv2.imwrite("uploaded/colors.jpg", colors_img[0]*255)

    generated = c.sess.run(c.generated_images, feed_dict={c.line_images: lines_img, c.color_images: colors_img})

    cv2.imwrite("uploaded/gen.jpg", generated[0]*255)

    return static_file("uploaded/gen.jpg",
                       root=".",
                       mimetype='image/jpg')

run(host="0.0.0.0", port=8000)
