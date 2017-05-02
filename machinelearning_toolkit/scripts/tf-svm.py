import tensorflow as tf
import pandas as pd
from tensorflow.contrib.learn.python.learn.estimators import svm
# categorical base columns
# class_of_worker = tf.contrib.layers.sparse_column_with_keys(
#     column_name='class_of_worker',
#     keys=[
#         'Not in universe', 'Federal government', 'Local government',
#         'Never worked', 'Private', 'Self-employed-incorporated',
#         'Self-employed-not incorporated', 'State government', 'Without pay'
#     ])
class_of_worker = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='class_of_worker', hash_bucket_size=1000)
# detailed_industry_recode = tf.contrib.layers.sparse_column_with_keys(
#     column_name='detailed_industry_recode',
#     keys=[
#         '0', '40', '44', '2', '43', '47', '48', '1', '11', '19', '24', '25',
#         '32', '33', '34', '35', '36', '37', '38', '39', '4', '42', '45', '5',
#         '15', '16', '22', '29', '31', '50', '14', '17', '18', '28', '3', '30',
#         '41', '46', '51', '12', '13', '21', '23', '26', '6', '7', '9', '49',
#         '27', '8', '10', '20'
#     ])
detailed_industry_recode = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_industry_recode', hash_bucket_size=1000)
# detailed_occupation_recode = tf.contrib.layers.sparse_column_with_keys(
#     column_name='detailed_occupation_recode',
#     keys=[
#         '0', '12', '31', '44', '19', '32', '10', '23', '26', '28', '29', '42',
#         '40', '34', '14', '36', '38', '2', '20', '25', '37', '41', '27', '24',
#         '30', '43', '33', '16', '45', '17', '35', '22', '18', '39', '3', '15',
#         '13', '46', '8', '21', '9', '4', '6', '5', '1', '11', '7'
#     ])
detailed_occupation_recode = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_occupation_recode', hash_bucket_size=1000)
# education = tf.contrib.layers.sparse_column_with_keys(
#     column_name='education',
#     keys=[
#         'Children', '7th and 8th grade', '9th grade', '10th grade',
#         'High school graduate', '11th grade', '12th grade no diploma',
#         '5th or 6th grade', 'Less than 1st grade',
#         'Bachelors degree(BA AB BS)', '1st 2nd 3rd or 4th grade',
#         'Some college but no degree', 'Masters degree(MA MS MEng MEd MSW MBA)',
#         'Associates degree-occup /vocational',
#         'Associates degree-academic program', 'Doctorate degree(PhD EdD)',
#         'Prof school degree (MD DDS DVM LLB JD)'
#     ])
education = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='education', hash_bucket_size=1000)
# enroll_in_edu_inst_last_wk = tf.contrib.layers.sparse_column_with_keys(
#     column_name='enroll_in_edu_inst_last_wk',
#     keys=[
#         'Not in universe', 'High school', 'College or university',
#     ])
enroll_in_edu_inst_last_wk = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='enroll_in_edu_inst_last_wk', hash_bucket_size=1000)
# marital_stat = tf.contrib.layers.sparse_column_with_keys(
#     column_name='marital_stat',
#     keys=[
#         'Never married', 'Married-civilian spouse present',
#         'Married-spouse absent', 'Separated', 'Divorced', 'Widowed',
#         'Married-A F spouse present'
#     ])
marital_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='marital_stat', hash_bucket_size=1000)
# major_industry_code = tf.contrib.layers.sparse_column_with_keys(
#     column_name='major_industry_code',
#     keys=[
#         'Not in universe or children', 'Entertainment', 'Social services',
#         'Agriculture', 'Education', 'Public administration',
#         'Manufacturing-durable goods', 'Manufacturing-nondurable goods',
#         'Wholesale trade', 'Retail trade', 'Finance insurance and real estate',
#         'Private household services', 'Business and repair services',
#         'Personal services except private HH', 'Construction',
#         'Medical except hospital', 'Other professional services',
#         'Transportation', 'Utilities and sanitary services', 'Mining',
#         'Communications', 'Hospital services', 'Forestry and fisheries',
#         'Armed Forces'
#     ])
major_industry_code = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='major_industry_code', hash_bucket_size=1000)
# major_occupation_code = tf.contrib.layers.sparse_column_with_keys(
#     column_name='major_occupation_code',
#     keys=[
#         'Not in universe', 'Professional specialty', 'Other service',
#         'Farming forestry and fishing', 'Sales',
#         'Adm support including clerical', 'Protective services',
#         'Handlers equip cleaners etc ', 'Precision production craft & repair',
#         'Technicians and related support',
#         'Machine operators assmblrs & inspctrs',
#         'Transportation and material moving', 'Executive admin and managerial',
#         'Private household services', 'Armed Forces'
#     ])
major_occupation_code = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='major_occupation_code', hash_bucket_size=1000)

# race = tf.contrib.layers.sparse_column_with_keys(
#     column_name='race',
#     keys=[
#         'White', 'Black', 'Other', 'Amer Indian Aleut or Eskimo',
#         'Asian or Pacific Islander'
#     ])
race = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='race', hash_bucket_size=1000)
# hispanic_origin = tf.contrib.layers.sparse_column_with_keys(
#     column_name='hispanic_origin',
#     keys=[
#         'Mexican (Mexicano)', 'Mexican-American', 'Puerto Rican',
#         'Central or South American', 'All other', 'Other Spanish', 'Chicano',
#         'Cuban', 'Do not know', 'NA'
#     ])
hispanic_origin = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='hispanic_origin', hash_bucket_size=1000)
sex = tf.contrib.layers.sparse_column_with_keys(
    column_name='sex', keys=['Female', 'Male'])

# member_of_labor_union = tf.contrib.layers.sparse_column_with_keys(
#     column_name='member_of_labor_union', keys=['Not in universe', 'No', 'Yes'])
member_of_labor_union = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='member_of_labor_union', hash_bucket_size=1000)
# reason_for_unemployment = tf.contrib.layers.sparse_column_with_keys(
#     column_name='reason_for_unemployment',
#     keys=[
#         'Not in universe', 'Re-entrant', 'Job loser - on layoff',
#         'New entrant', 'Job leaver', 'Other job loser'
#     ])
reason_for_unemployment = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='reason_for_unemployment', hash_bucket_size=1000)
# full_or_part_time_employment_stat = tf.contrib.layers.sparse_column_with_keys(
#     column_name='full_or_part_time_employment_stat',
#     keys=['Children or Armed Forces', 'Full-time schedules',
#     'Unemployed part- time', 'Not in labor force', 'Unemployed full-time',
#     'PT for non-econ reasons usually FT', 'PT for econ reasons usually PT',
#     'PT for econ reasons usually FT']
#     )
full_or_part_time_employment_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='full_or_part_time_employment_stat', hash_bucket_size=1000)
# tax_filer_stat = tf.contrib.layers.sparse_column_with_keys(
#     column_name='tax_filer_stat',
#     keys=[
#         'Nonfiler', 'Joint one under 65 & one 65+', 'Joint both under 65',
#         'Single', 'Head of household', 'Joint both 65+'
#     ])
tax_filer_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='tax_filer_stat', hash_bucket_size=1000)

# region_of_previous_residence = tf.contrib.layers.sparse_column_with_keys(
#     column_name='region_of_previous_residence',
#     keys=['Not in universe', 'South', 'Northeast', 'West', 'Midwest', 'Abroad'])
region_of_previous_residence = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='region_of_previous_residence', hash_bucket_size=1000)
# state_of_previous_residence = tf.contrib.layers.sparse_column_with_keys(
#     column_name='state_of_previous_residence',
#     keys=[
#         'Not in universe', 'Utah', 'Michigan', 'North Carolina',
#         'North Dakota', 'Virginia', 'Vermont', 'Wyoming', 'West Virginia',
#         'Pennsylvania', 'Abroad', 'Oregon', 'California', 'Iowa', 'Florida',
#         'Arkansas', 'Texas', 'South Carolina', 'Arizona', 'Indiana',
#         'Tennessee', 'Maine', 'Alaska', 'Ohio', 'Montana', 'Nebraska',
#         'Mississippi', 'District of Columbia', 'Minnesota', 'Illinois',
#         'Kentucky', 'Delaware', 'Colorado', 'Maryland', 'Wisconsin',
#         'New Hampshire', 'Nevada', 'New York', 'Georgia', 'Oklahoma',
#         'New Mexico', 'South Dakota', 'Missouri', 'Kansas', 'Connecticut',
#         'Louisiana', 'Alabama', 'Massachusetts', 'Idaho', 'New Jersey'
#     ])
state_of_previous_residence = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='state_of_previous_residence', hash_bucket_size=1000)
# detailed_household_and_family_stat = tf.contrib.layers.sparse_column_with_keys(
#     column_name='detailed_household_and_family_stat',
#     keys=[
#         'Child <18 never marr not in subfamily',
#         'Other Rel <18 never marr child of subfamily RP',
#         'Other Rel <18 never marr not in subfamily',
#         'Grandchild <18 never marr child of subfamily RP',
#         'Grandchild <18 never marr not in subfamily', 'Secondary individual',
#         'In group quarters', 'Child under 18 of RP of unrel subfamily',
#         'RP of unrelated subfamily', 'Spouse of householder', 'Householder',
#         'Other Rel <18 never married RP of subfamily',
#         'Grandchild <18 never marr RP of subfamily',
#         'Child <18 never marr RP of subfamily',
#         'Child <18 ever marr not in subfamily',
#         'Other Rel <18 ever marr RP of subfamily',
#         'Child <18 ever marr RP of subfamily', 'Nonfamily householder',
#         'Child <18 spouse of subfamily RP',
#         'Other Rel <18 spouse of subfamily RP',
#         'Other Rel <18 ever marr not in subfamily',
#         'Grandchild <18 ever marr not in subfamily',
#         'Child 18+ never marr Not in a subfamily',
#         'Grandchild 18+ never marr not in subfamily',
#         'Child 18+ ever marr RP of subfamily',
#         'Other Rel 18+ never marr not in subfamily',
#         'Child 18+ never marr RP of subfamily',
#         'Other Rel 18+ ever marr RP of subfamily',
#         'Other Rel 18+ never marr RP of subfamily',
#         'Other Rel 18+ spouse of subfamily RP',
#         'Other Rel 18+ ever marr not in subfamily',
#         'Child 18+ ever marr Not in a subfamily',
#         'Grandchild 18+ ever marr not in subfamily',
#         'Child 18+ spouse of subfamily RP',
#         'Spouse of RP of unrelated subfamily',
#         'Grandchild 18+ ever marr RP of subfamily',
#         'Grandchild 18+ never marr RP of subfamily',
#         'Grandchild 18+ spouse of subfamily RP'
#     ])
detailed_household_and_family_stat = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_household_and_family_stat', hash_bucket_size=1000)
# detailed_household_summary_in_household = tf.contrib.layers.sparse_column_with_keys(
#     column_name='detailed_household_summary_in_household',
#     keys=[
#         'Child under 18 never married', 'Other relative of householder',
#         'Nonrelative of householder', 'Spouse of householder', 'Householder',
#         'Child under 18 ever married', 'Group Quarters- Secondary individual',
#         'Child 18 or older'
#     ])
detailed_household_summary_in_household = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='detailed_household_summary_in_household',
    hash_bucket_size=1000)
# migration_code_change_in_msa = tf.contrib.layers.sparse_column_with_keys(
#     column_name='migration_code_change_in_msa',
#     keys=[
#         'Not in universe', 'Nonmover', 'MSA to MSA', 'NonMSA to nonMSA',
#         'MSA to nonMSA', 'NonMSA to MSA', 'Abroad to MSA', 'Not identifiable',
#         'Abroad to nonMSA','?'
#     ])
migration_code_change_in_msa = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_code_change_in_msa', hash_bucket_size=1000)
# migration_code_change_in_msa = tf.contrib.layers.sparse_column_with_keys(
#     column_name='migration_code_change_in_msa',
#     keys=[
#         'Not in universe', 'Nonmover', 'MSA to MSA', 'NonMSA to nonMSA',
#         'MSA to nonMSA', 'NonMSA to MSA', 'Abroad to MSA', 'Not identifiable',
#         'Abroad to nonMSA','?'
#     ])
migration_code_change_in_msa = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_code_change_in_msa', hash_bucket_size=1000)
# migration_code_change_in_reg = tf.contrib.layers.sparse_column_with_keys(
#     column_name='migration_code_change_in_reg',
#     keys=[
#         'Not in universe', 'Nonmover', 'Same county',
#         'Different county same state', 'Different state same division',
#         'Abroad', 'Different region', 'Different division same region'
#     ])
migration_code_change_in_reg = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_code_change_in_reg', hash_bucket_size=1000)
# migration_code_move_within_reg = tf.contrib.layers.sparse_column_with_keys(
#     column_name='migration_code_move_within_reg',
#     keys=[
#         'Not in universe', 'Nonmover', 'Same county',
#         'Different county same state', 'Different state in West', 'Abroad',
#         'Different state in Midwest', 'Different state in South',
#         'Different state in Northeast'
#     ])
migration_code_move_within_reg = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_code_move_within_reg', hash_bucket_size=1000)
# live_in_this_house_1year_ago = tf.contrib.layers.sparse_column_with_keys(
#     column_name='live_in_this_house_1year_ago',
#     keys=['Not in universe under 1 year old', 'Yes', 'No'])
live_in_this_house_1year_ago = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='live_in_this_house_1year_ago', hash_bucket_size=1000)
# migration_prev_res_in_sunbelt = tf.contrib.layers.sparse_column_with_keys(
#     column_name='migration_prev_res_in_sunbelt',
#     keys=['Not in universe', 'Yes', 'No'])
migration_prev_res_in_sunbelt = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='migration_prev_res_in_sunbelt', hash_bucket_size=1000)
# family_members_under18 = tf.contrib.layers.sparse_column_with_keys(
#     column_name='family_members_under18',
#     keys=[
#         'Both parents present', 'Neither parent present',
#         'Mother only present', 'Father only present', 'Not in universe'
#     ])
family_members_under18 = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='family_members_under18', hash_bucket_size=1000)
# country_of_birth_father = tf.contrib.layers.sparse_column_with_keys(
#     column_name='country_of_birth_father',
#     keys=[
#         'Mexico', 'United-States', 'Puerto-Rico', 'Dominican-Republic',
#         'Jamaica', 'Cuba', 'Portugal', 'Nicaragua', 'Peru', 'Ecuador',
#         'Guatemala', 'Philippines', 'Canada', 'Columbia', 'El-Salvador',
#         'Japan', 'England', 'Trinadad&Tobago', 'Honduras', 'Germany', 'Taiwan',
#         'Outlying-U S (Guam USVI etc)', 'India', 'Vietnam', 'China',
#         'Hong Kong', 'Cambodia', 'France', 'Laos', 'Haiti', 'South Korea',
#         'Iran', 'Greece', 'Italy', 'Poland', 'Thailand', 'Yugoslavia',
#         'Holand-Netherlands', 'Ireland', 'Scotland', 'Hungary', 'Panama'
#     ])
country_of_birth_father = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='country_of_birth_father', hash_bucket_size=1000)
# country_of_birth_mother = tf.contrib.layers.sparse_column_with_keys(
#     column_name='country_of_birth_mother',
#     keys=[
#         'India', 'Mexico', 'United-States', 'Puerto-Rico',
#         'Dominican-Republic', 'England', 'Honduras', 'Peru', 'Guatemala',
#         'Columbia', 'El-Salvador', 'Philippines', 'France', 'Ecuador',
#         'Nicaragua', 'Cuba', 'Outlying-U S (Guam USVI etc)', 'Jamaica',
#         'South Korea', 'China', 'Germany', 'Yugoslavia', 'Canada', 'Vietnam',
#         'Japan', 'Cambodia', 'Ireland', 'Laos', 'Haiti', 'Portugal', 'Taiwan',
#         'Holand-Netherlands', 'Greece', 'Italy', 'Poland', 'Thailand',
#         'Trinadad&Tobago', 'Hungary', 'Panama', 'Hong Kong', 'Scotland', 'Iran'
#     ])
country_of_birth_mother = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='country_of_birth_mother', hash_bucket_size=1000)
# country_of_birth_self = tf.contrib.layers.sparse_column_with_keys(
#     column_name='country_of_birth_self',
#     keys=[
#         'United-States', 'Mexico', 'Puerto-Rico', 'Peru', 'Canada',
#         'South Korea', 'India', 'Japan', 'Haiti', 'El-Salvador',
#         'Dominican-Republic', 'Portugal', 'Columbia', 'England',
#         'Thailand', 'Cuba', 'Laos', 'Panama', 'China', 'Germany',
#         'Vietnam', 'Italy', 'Honduras', 'Outlying-U S (Guam USVI etc)',
#         'Hungary', 'Philippines', 'Poland', 'Ecuador', 'Iran', 'Guatemala',
#         'Holand-Netherlands', 'Taiwan', 'Nicaragua', 'France', 'Jamaica',
#         'Scotland', 'Yugoslavia', 'Hong Kong', 'Trinadad&Tobago', 'Greece',
#         'Cambodia', 'Ireland'
#     ]
# )
country_of_birth_self = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='country_of_birth_self', hash_bucket_size=1000)
# citizenship = tf.contrib.layers.sparse_column_with_keys(
#     column_name='citizenship',
#     keys=[
#         'Native- Born in the United States',
#         'Foreign born- Not a citizen of U S ',
#         'Native- Born in Puerto Rico or U S Outlying',
#         'Native- Born abroad of American Parent(s)',
#         'Foreign born- U S citizen by naturalization'
#     ]
# )
citizenship = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='citizenship', hash_bucket_size=1000)

# own_business_or_self_employed = tf.contrib.layers.sparse_column_with_keys(
#     column_name='own_business_or_self_employed',
#     keys=['0', '2', '1']
# )
own_business_or_self_employed = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='own_business_or_self_employed', hash_bucket_size=1000)
# fill_inc_questionnaire_for_veteran_admin = tf.contrib.layers.sparse_column_with_keys(
#     column_name='fill_inc_questionnaire_for_veteran_admin',
#     keys=['Not in universe', 'Yes', 'No']
# )
fill_inc_questionnaire_for_veteran_admin = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='fill_inc_questionnaire_for_veteran_admin',
    hash_bucket_size=1000)

# veterans_benefits = tf.contrib.layers.sparse_column_with_keys(
#     column_name='veterans_benefits',
#     keys=['0', '2', '1']
# )
veterans_benefits = tf.contrib.layers.sparse_column_with_hash_bucket(
    column_name='veterans_benefits', hash_bucket_size=1000)
year = tf.contrib.layers.sparse_column_with_keys(
    column_name='year', keys=['94', '95'])
# label = tf.contrib.layers.sparse_column_with_keys(
#     column_name='label',
#     keys=['- 50000.', '50000+.']
# )

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
# print df_train.dtypes
dtypess = df_train.dtypes

# print dtypess[CATEGORICAL_COLUMNS]

# print df_train.head(5)
# print df_test.head(5)


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # # the values of that column stored in a constant Tensor.
    # continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
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
    # Add example id list
    feature_cols['example_id'] = tf.constant(
        [str(i + 1) for i in range(df['age'].size)])
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)


model_dir = '../svm_model_dir'

model = svm.SVM(example_id_column='example_id',
                feature_columns=FEATURE_COLUMNS,
                model_dir=model_dir)
model.fit(input_fn=train_input_fn, steps=100)
results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

# print list(set(df_train['migration_code_change_in_msa']))