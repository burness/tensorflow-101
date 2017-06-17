# -*- coding:utf-8 -*-
import sys
sys.path.append("../")
from config import *
import os
import logging
import operator
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def prepare_dataset():
    dataset_info = {}
    files = os.listdir(DATA_DIR)
    with open(all_text_filename, 'w') as fwrite:
        for file in files:
            i = 0
            file_path = os.path.join(DATA_DIR, file)
            logger.info("Process file {0}".format(file_path))
            with open(file_path, 'r') as fread:
                for line in fread.readlines():
                    i += 1
                    line_list = line.split("|")
                    # print line
                    if len(line_list) >= 3:
                        doc_text = line.split("|")[2]
                        w_line = str(filename_label[int(
                            file)]) + "\t" + doc_text
                        fwrite.write(w_line)
                dataset_info[file] = i
    print dataset_info
    sorted_dataset_info = sorted(
        dataset_info.items(), key=operator.itemgetter(1), reverse=True)
    print sorted_dataset_info


if __name__ == "__main__":
    prepare_dataset()
