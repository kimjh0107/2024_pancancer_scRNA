from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os


# def data_split_from_file(data_split_df, exp_df, crit):
#     target_samples = data_split_df.loc[lambda x:x[crit] == True, :].index
#     common_idx = sorted(list(set(target_samples).intersection(set(exp_df.index.map(lambda x: x[:-1])))))
#     sample_filt = exp_df.index.map(lambda x: True if x[:-1] in common_idx else False)
#     idx = exp_df.reset_index()[sample_filt]['index']
#     return idx

def data_split_from_file(data_split_df, exp_df, crit):

    target_samples = data_split_df.loc[lambda x:x[crit] == True, :].index
    common_idx = sorted(list(set(target_samples).intersection(set(exp_df.index.map(lambda x: x[:-1])))))
    sample_filt = exp_df.index.map(lambda x: True if x[:-1] in common_idx else False)
    
    # Instead of trying to access 'index' column, which may not exist, directly work with the filtered DataFrame
    filtered_df = exp_df.reset_index()[sample_filt]
    
    # If you need to return the index of the filtered DataFrame, you can directly return it
    idx = filtered_df.index
    
    return idx


def data_split(dataset, dict_patient_label, test_ratio=0.1, val_ratio=0.1):
    common_samples = list(set(dataset.index).intersection(dict_patient_label.keys()))
    dataset = dataset.loc[common_samples,:]
    labels = list(dataset.index.map(lambda x:dict_patient_label[x]))
    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=test_ratio, shuffle=True, stratify=labels)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, shuffle=True, stratify=[labels[idx] for idx in train_val_idx])

    return train_idx, val_idx, test_idx


def df_feature_scaling(df, train=True, scaler=None):
    if train == True:
        scaler = MinMaxScaler(feature_range=(0, 1))
        x = df.to_numpy()
        df_scaled = pd.DataFrame(scaler.fit_transform(
            x), index=df.index, columns=df.columns)
        return df_scaled, scaler
    else:
        x = df.to_numpy()
        x_scaled = scaler.transform(x)
        df_scaled = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)
        df_scaled = df_scaled.applymap(lambda x: 1 if x > 1 else x)
        df_scaled = df_scaled.applymap(lambda x: 0 if x < 0 else x)
        return df_scaled

def dataset_feature_scaling(df, train_idx, val_idx, test_idx):
    # common 요소 찾기 위해서 index reset. 하지 않으면 common idx 아무것도없음 
    df.reset_index(drop=True, inplace=True)

    train_idx_common = list(set(train_idx).intersection(set(df.index)))
    train_df, scaler = df_feature_scaling(
        df.loc[train_idx_common, :], train=True)

    val_idx_common = list(set(val_idx).intersection(set(df.index)))
    val_df = df_feature_scaling(
        df.loc[val_idx_common, :], train=False, scaler=scaler)

    test_idx_common = list(set(test_idx).intersection(set(df.index)))
    test_df = df_feature_scaling(
        df.loc[test_idx_common, :], train=False, scaler=scaler)
    return train_df, val_df, test_df


def df_2_list_pickle(df, dict_patient_label, path):
    # 디렉터리가 없으면 생성
    if not os.path.exists(path):
        os.makedirs(path)
    
    # DataFrame의 인덱스를 정렬하여 순서를 고정
    df_sorted = df.sort_index()
    
    li_GNNinput = []
    for row in df_sorted.index:
        # dict_patient_label에 row가 존재하는지 확인
        if row in dict_patient_label:
            # row가 존재하면 해당 데이터를 li_GNNinput에 추가
            exp_tmp = df_sorted.loc[[row], :]
            li_GNNinput.append(
                {'omics': exp_tmp, 'label': dict_patient_label[row]})
        else:
            # row가 존재하지 않으면 경고 메시지 출력(선택적)
            print(f"Warning: {row} is not in dict_patient_label.")

    # 리스트를 pickle 파일로 저장
    with open(os.path.join(path, "singleomics.pickle"), 'wb') as f:
        pickle.dump(li_GNNinput, f)

# commone한 환자들 이름만 들어가야지 인식이 가능한거


def df_3_list_pickle(df, dict_patient_label, path):
    if not os.path.exists(path):
        os.makedirs(path)
    li_GNNinput = []
    for idx, row in enumerate(df.index):
        for key, value in dict_patient_label.items():
            exp_tmp = df.loc[[row], :]
            li_GNNinput.append({'omics': exp_tmp, 'label': value})
    with open(path + "/singleomics.pickle", 'wb') as f:
        pickle.dump(li_GNNinput, f)


def df_2_list_pickle_multiomics(df_exp, df_prot, dict_patient_label, path):
    if not os.path.exists(path):
        os.makedirs(path)
    li_GNNinput = []
    for idx, row in enumerate(df_exp.index):
        if row in df_exp.index and row in df_prot.index:
            exp_tmp = df_exp.loc[[row], :]
            prot_tmp = df_prot.loc[[row], :]
            multiomics_tmp = pd.concat([exp_tmp, prot_tmp], axis=0)
            multiomics_tmp.fillna(0, inplace=True)
            li_GNNinput.append(
                {'multiomics': multiomics_tmp, 'label': dict_patient_label[row]})

    with open(path + "/multiomics.pickle", 'wb') as f:
        pickle.dump(li_GNNinput, f)
