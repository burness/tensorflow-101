import tensorflow as tf
import pandas as pd
from tensorflow.contrib.tensor_forest.client import random_forest

detailed_occupation_recode = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_occupation_recode', hash_bucket_size=1000)
education = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='education', hash_bucket_size=1000)
# Continuous base columns
age = tf.contrib.layers.real_valued_column('age')
wage_per_hour = tf.contrib.layers.real_valued_column('wage_per_hour')

columns = [
    'age', 'detailed_occupation_recode', 'education', 'wage_per_hour', 'label'
]
FEATURE_COLUMNS = [
    # age, age_buckets, class_of_worker, detailed_industry_recode,
    age,
    detailed_occupation_recode,
    education,
    wage_per_hour
]

LABEL_COLUMN = 'label'

CONTINUOUS_COLUMNS = ['age', 'wage_per_hour']

CATEGORICAL_COLUMNS = ['detailed_occupation_recode', 'education']

df_train = pd.DataFrame(
    [[12, '12', '7th and 8th grade', 40, '- 50000'],
     [40, '45', '7th and 8th grade', 40, '50000+'],
     [50, '50', '10th grade', 40, '50000+'],
     [60, '30', '7th and 8th grade', 40, '- 50000']],
    columns=[
        'age', 'detailed_occupation_recode', 'education', 'wage_per_hour',
        'label'
    ])

df_test = pd.DataFrame(
    [[12, '12', '7th and 8th grade', 40, '- 50000'],
     [40, '45', '7th and 8th grade', 40, '50000+'],
     [50, '50', '10th grade', 40, '50000+'],
     [60, '30', '7th and 8th grade', 40, '- 50000']],
    columns=[
        'age', 'detailed_occupation_recode', 'education', 'wage_per_hour',
        'label'
    ])
df_train[LABEL_COLUMN] = (
    df_train[LABEL_COLUMN].apply(lambda x: '+' in x)).astype(int)
df_test[LABEL_COLUMN] = (
    df_test[LABEL_COLUMN].apply(lambda x: '+' in x)).astype(int)
dtypess = df_train.dtypes

print df_train
print df_test


def input_fn(df):
    continuous_cols = {
        k: tf.expand_dims(tf.constant(df[k].values), 1)
        for k in CONTINUOUS_COLUMNS
    }
    # continuous_cols = {
    #     k: tf.constant(df[k].values)
    #     for k in CONTINUOUS_COLUMNS
    # }
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS
    }
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label


def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)


model_dir = '../rf_model_dir'

hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
    num_trees=10, max_nodes=1000, num_classes=2, num_features=4)
classifier = random_forest.TensorForestEstimator(hparams, model_dir=model_dir)
classifier.fit(input_fn=train_input_fn, steps=100)
results = classifier.evaluate(input_fn=eval_input_fn, steps=1)
print results
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
