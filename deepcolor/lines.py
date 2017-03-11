import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from random import randint

data = glob("imgs-valid/*.jpg")
for imname in data:

    cimg = cv2.imread(imname,1)
    cimg = np.fliplr(cimg.reshape(-1,3)).reshape(cimg.shape)
    cimg = cv2.resize(cimg, (256,256))

    img = cv2.imread(imname,0)

    # kernel = np.ones((5,5),np.float32)/25
    for i in xrange(30):
        randx = randint(0,205)
        randy = randint(0,205)
        cimg[randx:randx+50, randy:randy+50] = 255
    blur = cv2.blur(cimg,(100,100))


    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_edge = cv2.adaptiveThreshold(img, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)
    # img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    # img_cartoon = cv2.bitwise_and(img, img_edge)

    plt.subplot(131),plt.imshow(cimg)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132),plt.imshow(blur)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(133),plt.imshow(img_edge,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()
