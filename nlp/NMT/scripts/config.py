import re
import logging
import sys

NON_ALPHA_PAT = re.compile('[\.,-]')
COPUS_TYPE = "middle"
TRAIN_X = "train.en"
TRAIN_Y = "train.vi"
TEST_X_2012 = "tst2012.en"
TEST_Y_2012 = "tst2012.cs"
INPUR_SEQ_LENGTH = 30

logger = logging.getLogger("Neural Machine Translator")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
file_handler = logging.FileHandler("test.log")
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)