import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer

df_pre = pd.read_csv('../Data/aug_train.csv').drop('enrollee_id', axis = 1)

# df_pre.drop('enrollee_id', axis = 1)

df_pre.info()

print(df_pre['training_hours'].value_counts())


train_hrs_bins = [0, 50, 100, 150, 200, 250]
train_hrs_labels = [1, 2, 3, 4, 5]
df_pre['training_hours'] = pd.cut(x = df_pre['training_hours'], bins = train_hrs_bins, labels = train_hrs_labels, include_lowest = True)

print(df_pre['training_hours'].value_counts())



gender_map = {
    'Female': 2,
    'Male': 1,
    'Other': 0
}

relevent_experience_map = {
    'Has relevent experience': 1,
    'No relevent experience': 0
}

enrolled_university_map = {
    'no_enrollment': 0,
    'Full time course': 1,
    'Part time course': 2
}

education_level_map = {
    'Primary School': 0,
    'Graduate': 2,
    'Masters': 3,
    'High School': 1,
    'Phd': 4
}

major_map = {
    'STEM': 0,
    'Business Degree': 1,
    'Arts': 2,
    'Humanities': 3,
    'No Major': 4,
    'Other': 5
}

experience_map = {
    '<1': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10': 10,
    '11': 11,
    '12': 12,
    '13': 13,
    '14': 14,
    '15': 15,
    '16': 16,
    '17': 17,
    '18': 18,
    '19': 19,
    '20': 20,
    '>20': 21
}

company_type_map = {
    'Pvt Ltd': 0,
    'Funded Startup': 1,
    'Early Stage Startup': 2,
    'Other': 3,
    'Public Sector': 4,
    'NGO': 5
}

company_size_map = {
    '<10': 0,
    '10/49': 1,
    '100-500': 2,
    '1000-4999': 3,
    '10000+': 4,
    '50-99': 5,
    '500-999': 6,
    '5000-9999': 7
}

last_new_job_map = {
    'never': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '>4': 5
}



df_pre.loc[:,'education_level'] = df_pre['education_level'].map(education_level_map)
df_pre.loc[:,'company_size'] = df_pre['company_size'].map(company_size_map)
df_pre.loc[:,'company_type'] = df_pre['company_type'].map(company_type_map)
df_pre.loc[:,'last_new_job'] = df_pre['last_new_job'].map(last_new_job_map)
df_pre.loc[:,'major_discipline'] = df_pre['major_discipline'].map(major_map)
df_pre.loc[:,'enrolled_university'] = df_pre['enrolled_university'].map(enrolled_university_map)
df_pre.loc[:,'relevent_experience'] = df_pre['relevent_experience'].map(relevent_experience_map)
df_pre.loc[:,'gender'] = df_pre['gender'].map(gender_map)
df_pre.loc[:,'experience'] = df_pre['experience'].map(experience_map)






lb_en = LabelEncoder()

df_pre.loc[:,'city'] = lb_en.fit_transform(df_pre.loc[:,'city'])





# print(df_pre.value_counts())



df_pre1 = df_pre.copy()

prec_dict = {}

missing_cols = df_pre.columns[df_pre.isna().any()].tolist()

for miss_col in missing_cols:
    prec_dict[miss_col] = 0

knn_imputer = KNNImputer(n_neighbors = 7)

X = knn_imputer.fit_transform(df_pre1)
df_pre1 = pd.DataFrame(X, columns = df_pre1.columns)

df_pre1 = df_pre1.round(prec_dict)


all_cols = df_pre1.columns.values.tolist()
#
for col in all_cols:
    print(col)
    print(df_pre1[col].unique())
    print(len(df_pre1[col].unique()))

df_pre1.to_csv('../Data/processed_train.csv', index = False)