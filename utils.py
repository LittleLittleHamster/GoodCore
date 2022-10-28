import pandas as pd
import numpy as np
import datawig
import random
import imputation

def get_all_missing_position(df_cur):
    col_list = list(df_cur.columns)
    missing_dict = {}
    for i in range(0, len(df_cur)):
        for j in range(len(col_list)):
            if df_cur.iloc[i, j] == -1:
                if i not in missing_dict.keys():
                    missing_dict[i] = []
                missing_dict[i].append(j)
    return missing_dict


def get_prob(df_cur, missing_dict, label_col):
    mask_vec = df_cur.copy()
    mask_vec = mask_vec * 0  + 1
    for miss_tuple_id in missing_dict.keys():
        for col in missing_dict[miss_tuple_id]:
            mask_vec[col].iloc[miss_tuple_id] = 0

    col_list = list(df_cur.columns)
    col_list.remove(label_col)
    imputer = datawig.SimpleImputer(
        input_columns=col_list,
        output_column=label_col
    )
    imputed = imputer.predict_proba(df_cur)
    return imputed


def get_label_num(df_cur, label_col):
    label_num = df_cur[label_col].value_counts(normalize=True, ascending=True)
    return label_num




def select_init(df_train_dirty, df_train_prob, cs_size, sam_size, label_col):
    label_prop = get_label_num(df_train_dirty, label_col)
    label_list = list(label_prop.index)
    part_cs_size = []
    part_df_train_dirty = []
    for i in range(len(label_list)):
        part_cs_size.append(cs_size * label_prop[label_list[i]])

        df_tmp = df_train_dirty.loc[df_train_dirty[label_col] == label_list[i]]
        part_df_train_dirty.append(df_tmp)

    cs_list = []
    for i in range(len(label_list)):
        tmp_cs = cal_orders_weight(part_df_train_dirty[i], part_cs_size[i], df_train_prob, sam_size)
        cs_list.append(tmp_cs)

    df_cs = pd.concat(cs_list)
    return df_cs



def cal_orders_weight(df_train_part_dirty, part_cs_size, df_train_prob, sam_size):
    col_list = list(df_train_part_dirty.columns)
    col_list.append('ori_id')
    col_list.append('weight')
    df_cs_part = pd.DataFrame(columns=col_list)

    cs_or_not_list = range(0, len(df_train_part_dirty))

    for i in range(part_cs_size):
        sam_list = random.sample(cs_or_not_list, sam_size)
        d_utility_list = []
        ori_id_list = []
        wit_list = []

        for data_id in sam_list:
            d_utility, ori_id, wit = cal_utility(df_train_part_dirty, data_id,df_cs_part, df_train_prob)
            d_utility_list.append(d_utility)
            ori_id_list.append(ori_id)
            wit_list.append(wit)

        max_id = 0
        for j in range(len(sam_list)):
            if d_utility_list[j] >= d_utility_list[max_id]:
                max_id = j

        new_cs_tuple = list(df_train_part_dirty.iloc[sam_list[max_id]])
        new_cs_tuple.append(ori_id_list[max_id])
        new_cs_tuple.append(wit_list[max_id])
        df_cs_part.loc[len(df_cs_part)] = new_cs_tuple

        df_cs_part = imputation.human_imput(df_cs_part)


def cal_orders_weight_batch(df_train_part_dirty, part_cs_size, df_train_prob, sam_size, batch_size):
    col_list = list(df_train_part_dirty.columns)
    col_list.append('ori_id')
    col_list.append('weight')
    df_cs_part = pd.DataFrame(columns=col_list)

    cs_or_not_list = range(0, len(df_train_part_dirty))

    for i in range(part_cs_size):
        sam_list = random.sample(cs_or_not_list, sam_size)
        d_utility_list = []
        ori_id_list = []
        wit_list = []

        for data_id in sam_list:
            d_utility, ori_id, wit = cal_utility(df_train_part_dirty, data_id,df_cs_part, df_train_prob)
            d_utility_list.append(d_utility)
            ori_id_list.append(ori_id)
            wit_list.append(wit)

        max_id = 0
        for j in range(len(sam_list)):
            if d_utility_list[j] >= d_utility_list[max_id]:
                max_id = j

        new_cs_tuple = list(df_train_part_dirty.iloc[sam_list[max_id]])
        new_cs_tuple.append(ori_id_list[max_id])
        new_cs_tuple.append(wit_list[max_id])
        df_cs_part.loc[len(df_cs_part)] = new_cs_tuple

        if check_incomplete_num(df_cs_part) == batch_size:
            df_cs_part = imputation.human_imput(df_cs_part)

    if check_incomplete_num(df_cs_part) > 0:
        df_cs_part = imputation.human_imput(df_cs_part)


def check_complete(df_train_part_dirty, sam_id):
    col_list = list(df_train_part_dirty.columns)
    miss_cols = []
    for col in col_list:
        if df_train_part_dirty[col].iloc[sam_id] == -1:
            miss_cols.append(col)

    return miss_cols


def check_incomplete_num(df_cs):
    incomplete_n = 0
    for i in range(len(df_cs)):
        miss_col = check_complete(df_cs, i)
        if miss_col:
            incomplete_n += 1
    return incomplete_n


def cal_utility(df_train_part_dirty, sam_id, df_cs, df_train_prob):
    miss_col_list = check_complete(df_train_part_dirty, sam_id)
    col_list = list(df_train_part_dirty.columns)
    if len(miss_col_list) == 0:
        df_cs_tmp = df_cs.copy()
        new_tmp_cs_tuple = list(df_train_part_dirty.iloc[sam_id])
        df_cs_tmp.loc[len(df_cs_tmp)] = new_tmp_cs_tuple
        uti = find_min_cs(df_train_part_dirty, df_cs_tmp)
    else:
        df_pos = df_train_prob[df_train_prob.loc['org_index'] == sam_id]
        uti = 0
        for i in range(len(df_pos)):
            df_cs_tmp = df_cs.copy()
            new_tmp_cs_tuple = list(df_pos.iloc[i])
            df_cs_tmp.loc[len(df_cs_tmp)] = new_tmp_cs_tuple
            tmp_uti = find_min_cs(df_train_part_dirty, df_cs_tmp)
            uti += tmp_uti * df_pos['proba'].iloc[i]
    return uti


def find_min_cs(df_train_part_dirty, df_cs_tmp):
    tot_dis = 0
    for i in range(len(df_train_part_dirty)):
        min_cs_id = -1
        min_cs_dist = 1000000000
        for j in range(len(df_cs_tmp)):
            vec_train = df_train_part_dirty.loc[i].values
            vec_cs = df_cs_tmp.loc[j].values
            dist = np.sqrt(np.sum(np.square(vec_train - vec_cs)))

            if dist <= min_cs_dist:
                min_cs_id = j
                min_cs_dist = dist

        tot_dis += min_cs_dist

    return tot_dis



def LoadCoreset(file_name):
    dataset = np.load(f'{file_name}')
    order, weights = dataset['order'], dataset['weight']
    return order, weights

