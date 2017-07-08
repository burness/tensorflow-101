import tensorflow as tf
import pandas as pd
from tensorflow.contrib.learn.python.learn.estimators import svm

class_of_worker = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='class_of_worker', hash_bucket_size=1000)

detailed_industry_recode = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_industry_recode', hash_bucket_size=1000)

detailed_occupation_recode = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_occupation_recode', hash_bucket_size=1000)

education = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='education', hash_bucket_size=1000)

enroll_in_edu_inst_last_wk = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='enroll_in_edu_inst_last_wk', hash_bucket_size=1000)

marital_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='marital_stat', hash_bucket_size=1000)

major_industry_code = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='major_industry_code', hash_bucket_size=1000)

major_occupation_code = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='major_occupation_code', hash_bucket_size=1000)

race = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='race', hash_bucket_size=1000)

hispanic_origin = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='hispanic_origin', hash_bucket_size=1000)
sex = tf.contrib.layers.sparse_column_with_keys(
    column_name='sex', keys=['Female', 'Male'])

member_of_labor_union = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='member_of_labor_union', hash_bucket_size=1000)
reason_for_unemployment = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='reason_for_unemployment', hash_bucket_size=1000)

full_or_part_time_employment_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='full_or_part_time_employment_stat', hash_bucket_size=1000)

tax_filer_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='tax_filer_stat', hash_bucket_size=1000)

region_of_previous_residence = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='region_of_previous_residence', hash_bucket_size=1000)

state_of_previous_residence = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='state_of_previous_residence', hash_bucket_size=1000)

detailed_household_and_family_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_household_and_family_stat', hash_bucket_size=1000)

detailed_household_summary_in_household = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_household_summary_in_household',
    hash_bucket_size=1000)

migration_code_change_in_msa = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_code_change_in_msa', hash_bucket_size=1000)

migration_code_change_in_msa = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_code_change_in_msa', hash_bucket_size=1000)

migration_code_change_in_reg = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_code_change_in_reg', hash_bucket_size=1000)

migration_code_move_within_reg = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_code_move_within_reg', hash_bucket_size=1000)

live_in_this_house_1year_ago = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='live_in_this_house_1year_ago', hash_bucket_size=1000)

migration_prev_res_in_sunbelt = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_prev_res_in_sunbelt', hash_bucket_size=1000)

family_members_under18 = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='family_members_under18', hash_bucket_size=1000)

country_of_birth_father = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='country_of_birth_father', hash_bucket_size=1000)

country_of_birth_mother = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='country_of_birth_mother', hash_bucket_size=1000)

country_of_birth_self = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='country_of_birth_self', hash_bucket_size=1000)

citizenship = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='citizenship', hash_bucket_size=1000)

own_business_or_self_employed = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='own_business_or_self_employed', hash_bucket_size=1000)

fill_inc_questionnaire_for_veteran_admin = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='fill_inc_questionnaire_for_veteran_admin',
    hash_bucket_size=1000)

veterans_benefits = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='veterans_benefits', hash_bucket_size=1000)
year = tf.contrib.layers.sparse_column_with_keys(
    column_name='year', keys=['94', '95'])

# Continuous base columns
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

COLUMNS = [
    'age', 'class_of_worker', 'detailed_industry_recode',
    'detailed_occupation_recode', 'education', 'wage_per_hour',
    'enroll_in_edu_inst_last_wk', 'marital_stat', 'major_industry_code',
    'major_occupation_code', 'race', 'hispanic_origin', 'sex',
    'member_of_labor_union', 'reason_for_unemployment',
    'full_or_part_time_employment_stat', 'capital_gains', 'capital_losses',
    'dividends_from_stocks', 'tax_filer_stat', 'region_of_previous_residence',
    'state_of_previous_residence', 'detailed_household_and_family_stat',
    'detailed_household_summary_in_household', 'instance_weight',
    'migration_code_change_in_msa', 'migration_code_change_in_reg',
    'migration_code_move_within_reg', 'live_in_this_house_1year_ago',
    'migration_prev_res_in_sunbelt', 'num_persons_worked_for_employer',
    'family_members_under18', 'country_of_birth_father',
    'country_of_birth_mother', 'country_of_birth_self', 'citizenship',
    'own_business_or_self_employed',
    'fill_inc_questionnaire_for_veteran_admin', 'veterans_benefits',
    'weeks_worked_in_year', 'year', 'label'
]
FEATURE_COLUMNS = [
    age, age_buckets, class_of_worker, detailed_industry_recode,
    detailed_occupation_recode, education, wage_per_hour,
    enroll_in_edu_inst_last_wk, marital_stat, major_industry_code,
    major_occupation_code, race, hispanic_origin, sex, member_of_labor_union,
    reason_for_unemployment, full_or_part_time_employment_stat, capital_gains,
    capital_losses, dividends_from_stocks, tax_filer_stat,
    region_of_previous_residence, state_of_previous_residence,
    detailed_household_and_family_stat,
    detailed_household_summary_in_household, instance_weight,
    migration_code_change_in_msa, migration_code_change_in_reg,
    migration_code_move_within_reg, live_in_this_house_1year_ago,
    migration_prev_res_in_sunbelt, num_persons_worked_for_employer,
    family_members_under18, country_of_birth_father, country_of_birth_mother,
    country_of_birth_self, citizenship, own_business_or_self_employed,
    fill_inc_questionnaire_for_veteran_admin, veterans_benefits,
    weeks_worked_in_year, year
]

LABEL_COLUMN = 'label'
CONTINUOUS_COLUMNS = [
    'age', 'wage_per_hour', 'capital_gains', 'capital_losses',
    'dividends_from_stocks', 'instance_weight', 'weeks_worked_in_year',
    'num_persons_worked_for_employer'
]
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

TRAIN_FILE = '../data/census/census-income.data'
TEST_FILE = '../data/census/census-income.test'

df_train = pd.read_csv(TRAIN_FILE, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(TEST_FILE, names=COLUMNS, skipinitialspace=True)
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)
df_train[[
    'detailed_industry_recode', 'detailed_occupation_recode', 'year',
    'own_business_or_self_employed', 'veterans_benefits'
]] = df_train[[
    'detailed_industry_recode', 'detailed_occupation_recode', 'year',
    'own_business_or_self_employed', 'veterans_benefits'
]].astype(str)
df_test[[
    'detailed_industry_recode', 'detailed_occupation_recode', 'year',
    'own_business_or_self_employed', 'veterans_benefits'
]] = df_test[[
    'detailed_industry_recode', 'detailed_occupation_recode', 'year',
    'own_business_or_self_employed', 'veterans_benefits'
]].astype(str)

df_train[LABEL_COLUMN] = (
    df_train[LABEL_COLUMN].apply(lambda x: '+' in x)).astype(int)
df_test[LABEL_COLUMN] = (
    df_test[LABEL_COLUMN].apply(lambda x: '+' in x)).astype(int)
dtypess = df_train.dtypes


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # # the values of that column stored in a constant Tensor.
    continuous_cols = {
        k: tf.constant(df[k].values)
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


model_dir = '../model_dir'

model = tf.contrib.learn.LinearClassifier(
    feature_columns=FEATURE_COLUMNS, model_dir=model_dir)
model.fit(input_fn=train_input_fn, steps=200)
results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
