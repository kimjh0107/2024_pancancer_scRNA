import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import resample
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from ml_collections.config_dict import ConfigDict
import yaml


def load_pickle(path):
    with open(path, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


def create_dataframe(data):
    df_list = []
    for item in data:
        df = item['omics']
        df['label'] = item['label']
        df_list.append(df)
    return pd.concat(df_list, axis=0)


def main(TRAIN_PATH, VAL_PATH, TEST_PATH, OUTPUT_PATH):
    # 데이터 로드
    train_data = load_pickle(TRAIN_PATH)
    val_data = load_pickle(VAL_PATH)
    test_data = load_pickle(TEST_PATH)

    # DataFrame 생성
    train_df = create_dataframe(train_data)
    val_df = create_dataframe(val_data)
    test_df = create_dataframe(test_data)

    # 전체 데이터 합치기
    full_data = pd.concat([train_df, val_df, test_df])

    num_bootstraps = 100
    bootstrap_results = []
    cumulative_feature_importances = np.zeros(full_data.shape[1] - 1)

    for i in range(num_bootstraps):
        print('Num bootstrap:', i)
        sampled_data = resample(full_data, replace=True, n_samples=len(full_data))
        split_idx = int(0.8 * len(sampled_data))
        train_sampled = sampled_data[:split_idx]
        test_sampled = sampled_data[split_idx:]

        X_train = train_sampled.iloc[:, :-1]
        y_train = train_sampled.iloc[:, -1]
        y_train = y_train.astype(int)
        X_test = test_sampled.iloc[:, :-1]
        y_test = test_sampled.iloc[:, -1]
        y_test = y_test.astype(int)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train, verbose=False)

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        bootstrap_results.append((auroc, auprc))
        print(f"Bootstrap {i+1} - AUROC: {auroc}, AUPRC: {auprc}")

        cumulative_feature_importances += model.feature_importances_

    results_df = pd.DataFrame(bootstrap_results, columns=['AUROC', 'AUPRC'])
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Results saved to {OUTPUT_PATH}')

    average_feature_importances = cumulative_feature_importances / num_bootstraps
    important_features = [(feature, importance) for feature, importance in zip(
        full_data.columns[:-1], average_feature_importances) if importance > 0.001]

    FEATURE_PATH = Path(
        '/workspace/TCGA_pipeline/model/xgboost_bootstrapping/LUAD_TMB/LUAD_TMB_WGCNA_xgboost_bootstrap_results_features_importance.csv')
    with open(Path(FEATURE_PATH).with_suffix('.txt'), 'w') as file:
        for feature, _ in important_features:
            file.write(feature + '\n')


def main_feature(TRAIN_PATH, VAL_PATH, TEST_PATH, OUTPUT_PATH, FEATURE_PATH):
    train_data = load_pickle(TRAIN_PATH)
    val_data = load_pickle(VAL_PATH)
    test_data = load_pickle(TEST_PATH)

    train_df = create_dataframe(train_data)
    val_df = create_dataframe(val_data)
    test_df = create_dataframe(test_data)

    full_data = pd.concat([train_df, val_df, test_df])

    num_bootstraps = 100
    bootstrap_results = []
    cumulative_feature_importances = np.zeros(full_data.shape[1] - 1)

    for i in range(num_bootstraps):
        print('Num bootstrap:', i)
        sampled_data = resample(full_data, replace=True, n_samples=len(full_data))
        split_idx = int(0.8 * len(sampled_data))
        train_sampled = sampled_data[:split_idx]
        test_sampled = sampled_data[split_idx:]

        X_train = train_sampled.iloc[:, :-1]
        y_train = train_sampled.iloc[:, -1]
        y_train = y_train.astype(int)
        X_test = test_sampled.iloc[:, :-1]
        y_test = test_sampled.iloc[:, -1]
        y_test = y_test.astype(int)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train, verbose=False)

        y_pred_proba = model.predict_proba(X_test)

        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
            auroc = roc_auc_score(y_test, y_pred_proba)
        else:  # Multi-class classification
            auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

        auprc = average_precision_score(y_test, y_pred_proba)
        bootstrap_results.append((auroc, auprc))
        print(f"Bootstrap {i+1} - AUROC: {auroc}, AUPRC: {auprc}")

        cumulative_feature_importances += model.feature_importances_

    results_df = pd.DataFrame(bootstrap_results, columns=['AUROC', 'AUPRC'])
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Results saved to {OUTPUT_PATH}')

    average_feature_importances = cumulative_feature_importances / num_bootstraps
    important_features = [(feature, importance) for feature, importance in zip(
        full_data.columns[:-1], average_feature_importances) if importance > 0.001]

    with open(Path(FEATURE_PATH).with_suffix('.txt'), 'w') as file:
        for feature, importance in important_features:
            file.write(f"{feature}: {importance}\n")
    print(f'Important features saved to {FEATURE_PATH}')


if __name__ == '__main__':

    config = ConfigDict(
        yaml.load(open('/workspace/TCGA_pipeline/configs/tasks/REF_TCGA_downstream_task.yaml', 'r'), yaml.FullLoader))

    # TMB LUAD
    main_feature('/workspace/TCGA_pipeline/datasets/TMB_LUAD/hdWGCNA/TCGA-LUAD_tumor_suppressor_genes_split0/train/singleomics.pickle',
                 '/workspace/TCGA_pipeline/datasets/TMB_LUAD/hdWGCNA/TCGA-LUAD_tumor_suppressor_genes_split0/validation/singleomics.pickle',
                 '/workspace/TCGA_pipeline/datasets/TMB_LUAD/hdWGCNA/TCGA-LUAD_tumor_suppressor_genes_split0/test/singleomics.pickle',
                 '/workspace/TCGA_pipeline/model/xgboost_bootstrapping_manual/TMB_LUAD/TMB_LUAD_hdWGCNA_result_retest_tmb10.csv',
                 '/workspace/TCGA_pipeline/model/xgboost_bootstrapping_manual/TMB_LUAD/TMB_LUAD_hdWGCNA_features_retest_tmb10.csv')
