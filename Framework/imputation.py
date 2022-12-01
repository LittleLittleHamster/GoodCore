import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def human_imput(df_train_part_dirty):
	col_list = list(df_train_part_dirty.columns)
    miss_cols = []
    for i in range(len(df_train_part_dirty)):
	    for col in col_list:
	        if df_train_part_dirty[col].iloc[i] == -1:
	        	print(df_train_part_dirty.iloc[i])
	            df_train_part_dirty[col].iloc[i] = input("Please impute the missing value")

    return df_train_part_dirty


 def mice_imput(df_train_part_dirty):
 	imper = IterativeImputer(max_iter=10, random_state=0)
 	imp.fit(df_train_part_dirty)
 	df_train_part_clean = imp.transform(df_train_part_dirty)
 	return df_train_part_clean


