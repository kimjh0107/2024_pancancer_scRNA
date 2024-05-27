# python preprocessing/preprocessing.py -taskConfig ./configs/tasks/TCGA-COAD_MSI.yaml
# python preprocessing/preprocessing.py -taskConfig ./configs/tasks/TCGA-LUAD_TMB_seed42.yaml
# python preprocessing/preprocessing.py -taskConfig ./configs/tasks/TCGA-LUAD_TMB_valid.yaml

import pandas as pd
import networkx as nx
from utils_dataset import *
from glob import glob
from ml_collections.config_dict import ConfigDict
import yaml
import argparse
import random
import numpy as np


# ======== Prepare dataset ========
def prepare_dataset(config, multi_omics=False):
    exp_file = glob(config.data.path+'transcriptome_*')[0]
    exp_df = pd.read_csv(exp_file, sep='\t', index_col=0)

    prot_file = glob(config.data.path+'proteome_*')[0]
    prot_df = pd.read_csv(prot_file, sep='\t', index_col=0)

    patient_id_file = glob(config.data.path+'Patient_label_*')[0]
    dict_patient_id = pd.read_csv(
        patient_id_file, sep='\t', header=None, index_col=0).to_dict()[1]


    if os.path.isfile(config.data.data_split_file) == True:

        data_split_df = pd.read_csv(
            config.data.data_split_file, sep='\t', index_col=0)
        train_idx = data_split_from_file(data_split_df, exp_df, 'train')
        val_idx = data_split_from_file(data_split_df, exp_df, 'val')
        test_idx = data_split_from_file(data_split_df, exp_df, 'test')
        print("Train: {}, Val: {}, Test: {}".format(
            len(train_idx), len(val_idx), len(test_idx)))

    else:

        train_idx, val_idx, test_idx = data_split(exp_df, dict_patient_id)
        print("Train: {}, Val: {}, Test: {}".format(
            len(train_idx), len(val_idx), len(test_idx)))

        exp_df.index
        common_samples = sorted(list(set(exp_df.index).intersection(dict_patient_id.keys())))
        dataset = exp_df.loc[common_samples, :]

        split_df = pd.DataFrame(index=dataset.index)

        # save
        split_df['ID'] = split_df.index
        split_df['train'] = False
        split_df['val'] = False
        split_df['test'] = False

        # Update the DataFrame based on the indices
        split_df.loc[dataset.index[train_idx], 'train'] = True
        split_df.loc[dataset.index[val_idx], 'val'] = True
        split_df.loc[dataset.index[test_idx], 'test'] = True
        # ID를 조정합니다.
        split_df['ID'] = split_df['ID'].map(
            lambda x: x[:-1] if x.endswith('A') else x)

        split_df.to_csv(config.data.data_split_file, sep='\t', index=False)

    # Feature scaling
    train_exp, val_exp, test_exp = dataset_feature_scaling(exp_df, train_idx, val_idx, test_idx)

    exp_file = glob(config.data.path+'transcriptome_*')[0]
    exp_df = pd.read_csv(exp_file, sep='\t', index_col=0)
    index_to_name_mapping = {i: name for i, name in enumerate(exp_df.index)}

    train_exp.index = train_exp.index.map(index_to_name_mapping)
    val_exp.index = val_exp.index.map(index_to_name_mapping)
    test_exp.index = test_exp.index.map(index_to_name_mapping)


    if multi_omics == False:
        df_2_list_pickle(train_exp, dict_patient_id,
                         path=config.data.path+"train")
        df_2_list_pickle(val_exp, dict_patient_id,
                         path=config.data.path+"validation")
        df_2_list_pickle(test_exp, dict_patient_id,
                         path=config.data.path+"test")
    else:
        pass
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-taskConfig")
    args = parser.parse_args()
    config_dataset = ConfigDict(
        yaml.load(open(args.taskConfig, 'r'), yaml.FullLoader))

    with open(config_dataset.data.gene_list) as f:
        gene_list = f.read().strip().split('\n')

    multi_omics = config_dataset.data.omics_data
    for omics in multi_omics:
        exp = pd.read_csv(multi_omics[omics], sep='\t', index_col=0)
        common_genes = [gene for gene in gene_list if gene in exp.columns]
        exp_filtered = exp.loc[:, common_genes]
        exp_filtered.to_csv(config_dataset.data.path + f"/{omics}_Genefilt.tsv", sep='\t')
        
    prepare_dataset(config_dataset, multi_omics=True)
    prepare_dataset(config_dataset, multi_omics=False)



