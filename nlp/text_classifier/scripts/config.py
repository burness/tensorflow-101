import os
DATA_DIR = "../../data/origin_data"
DATA_DIR = os.path.abspath(DATA_DIR)
print DATA_DIR
filename = [
    100,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    115,
    116,
    118,
    119,
    121,
    122,
    123,
    124,
    148,
]
all_text_filename = os.path.join(DATA_DIR, 'all.csv')
filename_label = dict(zip(filename, range(len(filename))))
# W2V_FILE = "../../data/pretrain_w2v/zh/test.tsv"
W2V_FILE = "/Users/burness/git_repository/dl_opensource/nlp/oxford-cs-deepnlp-2017/practical-2/data/pretrain_w2v/w2vgood_20170209.model"
# W2V_FILE = os.path.abspath(W2V_FILE)
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 1500
CLASS_NUM = 2
WORD_DICT = "/Users/burness/git_repository/dl_opensource/nlp/oxford-cs-deepnlp-2017/practical-2/data/origin_data/t_tag_infos.txt"
all_title_filename = os.path.join(DATA_DIR, 'all_title.csv')
MAX_TITLE_LENGTH = 20
