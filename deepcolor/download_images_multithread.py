import os
import Queue
from threading import Thread
from time import time
from itertools import chain
import urllib2
import untangle
import numpy as np
import cv2

def download_imgs(url):
    # count = 0
    maxsize = 512
    file_name = url.split('=')[-1]
    header = {'Referer':'http://safebooru.org/index.php?page=post&s=list','User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
    request = urllib2.Request(url, headers=header)
    stringreturn = urllib2.urlopen(request).read()
    xmlreturn = untangle.parse(stringreturn)
    count = 0
    print xmlreturn.posts[0]['sample_url']
    try:
        for post in xmlreturn.posts.post:
            try:
                imgurl = "http:" + post["sample_url"]
                print imgurl
                if ("png" in imgurl) or ("jpg" in imgurl):
                    resp = urllib2.urlopen(imgurl)
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    height, width = image.shape[:2]
                    if height > width:
                        scalefactor = (maxsize*1.0) / width
                        res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                        cropped = res[0:maxsize,0:maxsize]
                    if width >= height:
                        scalefactor = (maxsize*1.0) / height
                        res = cv2.resize(image,(int(width * scalefactor), int(height*scalefactor)), interpolation = cv2.INTER_CUBIC)
                        center_x = int(round(width*scalefactor*0.5))
                        print center_x
                        cropped = res[0:maxsize,center_x - maxsize/2:center_x + maxsize/2]
                    count += 1
                    cv2.imwrite("imgs-valid/"+file_name+'_'+str(count)+'.jpg',cropped)
            except:
                continue
    except:
        print "no post in xml"
        return

class DownloadWorker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            url = self.queue.get()
            if url is None:
                break
            # download_link(directory, link)
            download_imgs(url)
            self.queue.task_done()

if __name__ == '__main__':
    start = time()
    download_queue = Queue.Queue(maxsize=100)
    for x in range(8):
        worker = DownloadWorker(download_queue)
        worker.daemon = True
        worker.start()
    
    url_links = ["http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=1girl%20solo&pid="+str(i+5000) for i in xrange(299,10000)]
    # print url_links[:10]

    for link in url_links:
        download_queue.put(link)
    download_queue.join()
    print "the images num is {0}".format(len(url_links))
    print "took time : {0}".format(time() - start)



