# 本项目是针对Oxford的文本分类作业的扩展，想拿下中文数据来做下分类

## 数据集介绍
下面数据集是真实的文本，31类，每一个文本内有若干新闻条数，文件名即为其分类，由于其中有几个类别文件数太少，所以只选择其中20类来做实验
    ├── origin_data
    │   ├── 100
    │   ├── 101
    │   ├── 103
    │   ├── 104
    │   ├── 105
    │   ├── 106
    │   ├── 107
    │   ├── 108
    │   ├── 109
    │   ├── 110
    │   ├── 111
    │   ├── 112
    │   ├── 113
    │   ├── 114
    │   ├── 115
    │   ├── 116
    │   ├── 117
    │   ├── 118
    │   ├── 119
    │   ├── 121
    │   ├── 122
    │   ├── 123
    │   ├── 124
    │   ├── 125
    │   ├── 126
    │   ├── 128
    │   ├── 141
    │   ├── 142
    │   ├── 143
    │   ├── 145
    │   └── 148

### 数据集处理
将文本数据内容读出，转换为label+"\t"+doc_text的一条条记录存储在all.csv,基本情况如下：

    [('106', 22840), ('105', 16624), ('109', 15589), ('107', 15094), ('108', 14963), ('123', 14353), ('115', 14295), ('116', 13980), ('110', 13911), ('103', 11834), ('104', 11500),
    ('118', 11370), ('119', 11274), ('112', 9965), ('100', 9102), ('122', 8150), ('148', 8076), ('111', 7652), ('124', 6210), ('121', 6051), ('117', 5369), ('101', 4454), ('113', 43
10), ('125', 3903), ('114', 3753), ('142', 2326), ('141', 1260), ('128', 901), ('126', 547), ('145', 381), ('143', 97)]

后面记录比较少，所以删除了记录数比较小的文件，只保留了20类


### Bow

#### Bow 原理

#### 结果

    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/0x/0m_rvxfs5dv2g90m_m2cdv7m0000gn/T/jieba.cache
    Loading model cost 0.441 seconds.
    Prefix dict has been built succesfully.
    all token len 1078806
    before filter, the tokens len: 1078806
    After filter, the tokens len: 196236
    dictionary shape (242588,196236)
    bow_sparse matrix shape:
    (242588, 196236)
    train set shape:
    (194070, 196236)
    Beigin to Train the model
    End Now, and evalution the model with test dataset
    mean accuracy: 0.989756379076
    save the trained model to lr_model.pl
    python bow_text_classifier.py  7302.72s user 238.53s system 66% cpu 3:08:43.57 total

### TF-IDF

TF-IDF 感觉就是在Bow的前提下，考虑每一个词的权重，而不仅仅是用count来作为矩阵当中的值
结果：

    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/0x/0m_rvxfs5dv2g90m_m2cdv7m0000gn/T/jieba.cache
    Loading model cost 0.398 seconds.
    Prefix dict has been built succesfully.
    all token len 1078806
    before filter, the tokens len: 1078806
    word len: 391724
    tfidf shape (242588,391724)
    train set shape:
    (194070, 391724)
    Beigin to Train the model
    End Now, and evalution the model with test dataset
                precision    recall  f1-score   support

            0       0.95      0.97      0.96      1823
            1       0.98      0.99      0.99      2352
            10       0.96      0.96      0.96      1939
            11       0.99      1.00      0.99      2819
            12       0.98      1.00      0.99      2827
            13       0.99      0.99      0.99      2226
            14       1.00      0.99      0.99      2321
            15       0.99      0.96      0.98      1166
            16       0.99      0.99      0.99      1662
            17       0.99      0.99      0.99      2906
            18       0.99      0.98      0.98      1227
            19       0.97      0.94      0.95      1643
            2       0.99      0.98      0.98      2353
            3       1.00      1.00      1.00      3300
            4       1.00      1.00      1.00      4554
            5       0.99      1.00      0.99      3020
            6       1.00      0.99      0.99      3000
            7       0.99      0.99      0.99      3095
            8       0.98      0.99      0.99      2755
            9       1.00      0.98      0.99      1530

    avg / total       0.99      0.99      0.99     48518

    [[1760   18    3   13    2    0    0    3    2    6    1    1    4    1
        0    1    2    2    3    1]
    [   3 2327    1    0    0    0    0    0    0    3    0   13    1    0
        1    1    0    2    0    0]
    [   7    1 1853   12    0    9    2    0    1    2    2   36    2    0
        1    3    0    2    6    0]
    [   5    0    2 2811    0    0    0    0    0    0    1    0    0    0
        0    0    0    0    0    0]
    [   1    0    1    0 2813    2    1    0    3    4    1    0    0    0
        0    0    0    1    0    0]
    [   0    0    3    2    9 2207    2    0    0    0    0    0    0    0
        1    0    0    1    0    1]
    [   0    0    2    0    0    1 2307    0    1    0    0    0    0    0
        0    0    3    6    0    1]
    [   2    0    6    0    3    0    0 1120    3    1    4    0    1    0
        0    3    0    5   16    2]
    [   0    0    1    0   10    2    0    0 1640    0    0    1    4    0
        0    1    0    1    2    0]
    [   3    1    4    3    3    0    0    0    0 2890    0    0    0    0
        0    1    0    0    1    0]
    [   2    0    2    1   12    1    0    0    2    0 1197    1    0    1
        0    5    0    0    3    0]
    [   3   17   51    2    1    2    1    0    0    4    0 1549    0    0
        0    4    3    0    6    0]
    [  32    2    0    0    0    2    0    1    1    0    0    0 2301    1
        0    0    4    7    2    0]
    [   2    0    0    0    0    0    0    0    0    0    0    0    0 3291
        2    2    0    3    0    0]
    [   0    0    0    0    0    0    0    0    0    0    1    0    0    0
    4549    2    0    1    1    0]
    [   0    1    1    0    0    0    0    0    1    0    0    1    0    0
        0 3014    0    1    0    1]
    [  16    0    1    2    0    0    3    1    0    0    0    0    3    0
        0    2 2967    4    1    0]
    [   2    0    2    0    0    1    1    0    0    2    0    0    4    8
        0    1    0 3072    1    1]
    [   6    0    4    0    0    0    0    3    0    0    0    0    1    0
        0    2    0    2 2737    0]
    [   0    0    3    1    5    2    1    0    1    1    0    0    0    1
        1   10    0    5    0 1499]]
    save the trained model to tfidf_lr_model.pl
    python tfidf_text_classifier.py  5469.91s user 25.33s system 10% cpu 14:45:45.01 total