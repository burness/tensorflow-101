import urllib2
import urllib
import json
import numpy as np
import cv2
import untangle

maxsize = 512

# tags = ["asu_tora","puuakachan","mankun","hammer_%28sunset_beach%29",""]

# for tag in tags:

count = 0

for i in xrange(299,10000):
    header = {'Referer':'http://safebooru.org/index.php?page=post&s=list','User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    url = "http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=1girl%20solo&pid="+str(i+5000)
    request = urllib2.Request(url, headers=header)
    stringreturn = urllib2.urlopen(request).read()
    print stringreturn
    xmlreturn = untangle.parse(stringreturn)
    for post in xmlreturn.posts.post:
        imgurl = "http:" + post["sample_url"]
        print imgurl
        if ("png" in imgurl) or ("jpg" in imgurl):

            resp = urllib.urlopen(imgurl)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            if height > width:
                scalefactor = (maxsize*1.0) / width
                res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                cropped = res[0:maxsize,0:maxsize]
            if width > height:
                scalefactor = (maxsize*1.0) / height
                res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                center_x = int(round(width*scalefactor*0.5))
                print center_x
                cropped = res[0:maxsize,center_x - maxsize/2:center_x + maxsize/2]

            # img_edge = cv2.adaptiveThreshold(cropped, 255,
            #                                  cv2.ADAPTIVE_THRESH_MEAN_C,
            #                                  cv2.THRESH_BINARY,
            #                                  blockSize=9,
            #                                  C=2)

            count += 1
            cv2.imwrite("imgs-valid/"+str(count)+".jpg",cropped)
            # cv2.imwrite("imgs/"+str(post["id"])+"-edge.jpg",img_edge)
