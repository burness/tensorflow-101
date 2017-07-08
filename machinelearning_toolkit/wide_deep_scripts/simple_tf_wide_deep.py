import tensorflow as tf
import pandas as pd
from tensorflow.contrib.learn.python.learn.estimators import svm
tf.logging.set_verbosity(tf.logging.INFO)


# logger = logging.getLogger('Training a classifier using wide and/or deep method')
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# logger.addHandler(ch)


tf.app.flags.DEFINE_string('classifier_mode', 'wide', 'Running mode. One of {"wide", "deep", "all"}')
tf.app.flags.DEFINE_integer('train_steps', 200, 'the step of train the model')
tf.app.flags.DEFINE_string('model_dir', '../wide_model_dir', 'the model dir')
FLAGS = tf.app.flags.FLAGS


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

# print df_train.dtypes
# dtypes = df_train.dtypes

# print dtypess[CATEGORICAL_COLUMNS]

# print df_train.head(5)
# print df_test.head(5)


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # # the values of that column stored in a constant Tensor.
    continuous_cols = {
        k: tf.expand_dims(tf.constant(df[k].values), 1)
        for k in CONTINUOUS_COLUMNS
    }
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS
    }
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)



# deep columns
"""
age = tf.contrib.layers.real_valued_column('age')
age_buckets = tf.contrib.layers.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
wage_per_hour = tf.contrib.layers.real_valued_column('wage_per_hour')
capital_gains = tf.contrib.layers.real_valued_column('capital_gains')
capital_losses = tf.contrib.layers.real_valued_column('capital_losses')
dividends_from_stocks = tf.contrib.layers.real_valued_column(
    'dividends_from_stocks')
instance_weight = tf.contrib.layers.real_valued_column('instance_weight')
weeks_worked_in_year = tf.contrib.layers.real_valued_column(
    'weeks_worked_in_year')
num_persons_worked_for_employer = tf.contrib.layers.real_valued_column(
    'num_persons_worked_for_employer')


CATEGORICAL_COLUMNS = [
    'class_of_worker', 'detailed_industry_recode',
    'detailed_occupation_recode', 'education', 'enroll_in_edu_inst_last_wk',
    'marital_stat', 'major_industry_code', 'major_occupation_code', 'race',
    'hispanic_origin', 'sex', 'member_of_labor_union',
    'reason_for_unemployment', 'full_or_part_time_employment_stat',
    'tax_filer_stat', 'region_of_previous_residence',
    'state_of_previous_residence', 'detailed_household_and_family_stat',
    'detailed_household_summary_in_household', 'migration_code_change_in_msa',
    'migration_code_change_in_reg', 'migration_code_move_within_reg',
    'live_in_this_house_1year_ago', 'migration_prev_res_in_sunbelt',
    'family_members_under18', 'country_of_birth_father',
    'country_of_birth_mother', 'country_of_birth_self', 'citizenship',
    'own_business_or_self_employed',
    'fill_inc_questionnaire_for_veteran_admin', 'veterans_benefits', 'year'
]
"""
wide_columns = [detailed_occupation_recode, education]
deep_columns = [age, wage_per_hour]

model_dir = FLAGS.model_dir
train_step = FLAGS.train_steps
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn = eval_input_fn,
    every_n_steps=100,
    eval_steps=1)
if FLAGS.classifier_mode == 'wide':
    model = tf.contrib.learn.LinearClassifier(model_dir=model_dir, 
        feature_columns=wide_columns, config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60))
elif FLAGS.classifier_mode == 'deep':
    model = tf.contrib.learn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[128, 64], config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
else:
    model = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[128, 64],
        fix_global_step_increment_bug=True,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60))

model.fit(input_fn=train_input_fn, steps=train_step, monitors=[validation_monitor])
results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in results:
    print "%s: %s" % (key, results[key])