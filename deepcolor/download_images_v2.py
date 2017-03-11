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
    def get_links(self) :
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
    
    url_links = ["http://safebooru.org/index.php?page=post&s=view&id={0}".format(string(i)) for i in xrange(299,10000)]
    # print url_links[:10]

    for link in url_links:
        download_queue.put(link)
    download_queue.join()
    print "the images num is {0}".format(len(url_links))
    print "took time : {0}".format(time() - start)



