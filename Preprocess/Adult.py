import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


from matplotlib import pyplot

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, log_loss, plot_roc_curve, auc, precision_recall_curve



train_data = '../Dataset/Adult/adult.data'
test_data = '../Dataset/Adult/adult.test'

columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus',
           'Occupation','Relationship','Race','Sex','CapitalGain',
           'CapitalLoss','HoursPerWeek','Country','Income']
df_train_set = pd.read_csv(train_data, names=columns)

df_test_set = pd.read_csv(test_data, names=columns, skiprows=1)



for i in df_train_set.columns:
    print(f"The column {i}'s dtype is {df_train_set.loc[:, i].dtype}")


object_columns = df_train_set.dtypes=="object"
object_columns = list(object_columns[object_columns].index)
int_columns = df_train_set.dtypes=="int64"
int_columns = list(int_columns[int_columns].index)


# Delete whitespace and dot
for col in df_train_set.columns:
    if df_train_set[col].dtype != 'int64':
        df_train_set[col] = df_train_set[col].apply(lambda val: val.replace(" ", ""))
        df_train_set[col] = df_train_set[col].apply(lambda val: val.replace(".", ""))
        df_test_set[col] = df_test_set[col].apply(lambda val: val.replace(" ", ""))
        df_test_set[col] = df_test_set[col].apply(lambda val: val.replace(".", ""))


df_train_set_clean = df_train_set.copy()
df_test_set_clean = df_test_set.copy()


# Replace ‘？’ with 'Unknown' in Full dataset

for i in df_train_set.columns:
    df_train_set.replace('?', 'Unknown', inplace=True)
    df_test_set.replace('?', 'Unknown', inplace = True)


df_train_set['target'] = 1
df_test_set['target'] = -1

df_pre = pd.concat([df_train_set, df_test_set], axis = 0).reset_index(drop = True)

df_pre.info()




df_train_set_clean['target'] = 1
df_test_set_clean['target'] = -1

df_pre_clean = pd.concat([df_train_set_clean, df_test_set_clean], axis = 0).reset_index(drop = True)

df_pre_clean.info()




# Object -> int
mapper = DataFrameMapper([('Workclass', LabelEncoder()),('Education', LabelEncoder()),
                          ('MaritalStatus', LabelEncoder()),('Occupation', LabelEncoder()),
                          ('Relationship', LabelEncoder()),('Race', LabelEncoder()),
                          ('Sex', LabelEncoder()),('Country', LabelEncoder()),
                          ('Income', LabelEncoder())], df_out=True, default=None)

df_pre = mapper.fit_transform(df_pre.copy())
df_pre_clean = mapper.fit_transform(df_pre_clean.copy())



# Continuous -> [0,1]
for col_name in int_columns:
    Scaler = MinMaxScaler(feature_range=(0, 1))
    col_value = np.array(df_pre[col_name]).reshape(-1,1)
    new_col = Scaler.fit_transform(col_value)
    df_pre[col_name] = new_col
    
    Scaler = MinMaxScaler(feature_range=(0, 1))
    col_value = np.array(df_pre_clean[col_name]).reshape(-1,1)
    new_col = Scaler.fit_transform(col_value)
    df_pre_clean[col_name] = new_col


df_train = df_pre[df_pre['target'] == 1].reset_index(drop = True).drop(['target'], axis = 1)
df_test = df_pre[df_pre['target'] == -1].reset_index(drop = True).drop(['target'], axis = 1)

df_train_clean = df_pre_clean[df_pre_clean['target'] == 1].reset_index(drop = True).drop(['target'], axis = 1)
df_test_clean = df_pre_clean[df_pre_clean['target'] == -1].reset_index(drop = True).drop(['target'], axis = 1)



df_train.to_csv("../Dataset/Adult/adult_train_all.csv", index = False)
df_test.to_csv("../Dataset/Adult/adult_test_all.csv", index = False)

df_train_clean.to_csv("../Dataset/Adult/adult_train_clean.csv", index = False)
df_test_clean.to_csv("../Dataset/Adult/adult_test_clean.csv", index = False)