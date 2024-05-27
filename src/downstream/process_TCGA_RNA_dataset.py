import os
import pandas as pd
import numpy as np 
from config import *


def process_TCGA_RNA_data(barcode_mapping_path, data_dir, gene_length_file_path):
    barcode_mapping = pd.read_csv(barcode_mapping_path, sep='\t')

    for uuid in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, uuid)
        if os.path.isdir(folder_path):
            barcode_row = barcode_mapping[barcode_mapping['uuid'] == uuid]

            if barcode_row.empty:
                continue

            barcode = barcode_row['submitter_id'].values[0]

            new_file_path = os.path.join(data_dir, f"{barcode}.tsv")

            with open(new_file_path, 'w') as f_out:
                f_out.write("gene\tgeneSymbol\treadCounts\n")

            for file in os.listdir(folder_path):            # 지정 경로 모든 항목 목록 받고
                # 폴더 경로, 파일 이름 결합해 전체 파일경로 생성
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):               # 생성된 경로 실제 파일인지 확인
                    with open(file_path, 'r') as f_in:
                        lines = f_in.readlines()
                        for line in lines[6:]:
                            parts = line.strip().split('\t')
                            if len(parts) >= 4:
                                with open(new_file_path, 'a') as f_out:
                                    f_out.write(
                                        f"{parts[0]}\t{parts[1]}\t{parts[3]}\n")

            # Normalization process
            df_inp = pd.read_csv(new_file_path, sep='\t', header=0).dropna()
            df_length = pd.read_csv(gene_length_file_path, sep='\t')
            dict_length = dict(zip(df_length['gene'], df_length['merged']))

            df_readCounts = df_inp.loc[:, [
                'gene', 'readCounts']].set_index('gene')

            # Calculate TMB
            df_TMB = (df_readCounts['readCounts'] /
                      df_readCounts.index.map(dict_length)).dropna()
            df_TMB = df_TMB / df_TMB.sum() * 10**6
            df_TMB = df_TMB.to_frame('TMB')

            df_result = df_inp.set_index('gene').join(df_TMB)

            df_result.reset_index().to_csv(new_file_path, sep='\t', index=False)


def merge_TCGA_data(data_dir, save_dir, cancer_type):
    output_filename = f"TMB_{cancer_type}_RNA_merged.tsv"
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(
        data_dir) if f.startswith("TCGA-") and f.endswith(".tsv")]

    dfs_list = []
    for file_path in file_paths:
        TCGA_barcode = os.path.basename(file_path).split('.')[0]
        df_tmp = pd.read_csv(file_path, sep='\t').loc[:, ['geneSymbol', 'TMB']].set_index(
            'geneSymbol').rename(columns={'TMB': TCGA_barcode})
        dfs_list.append(df_tmp)

    exp_merged = pd.concat(dfs_list, axis=1).T
    exp_merged = exp_merged.loc[:, exp_merged.sum() != 0]
    exp_merged.dropna(inplace=True)

    output_path = os.path.join(save_dir, output_filename)
    exp_merged.to_csv(output_path, sep='\t')

    # log transformation 
    exp_merged_scaled = np.log10(exp_merged + 1)
    output_path_scaled = os.path.join(save_dir, f"TMB_{cancer_type}_RNA_merged_logP_norm.tsv")
    exp_merged_scaled.to_csv(output_path_scaled, sep='\t')
    

def main(): 

    # PAAD TCGA RNA seq
    BARCODE_MAPPING_PATH = '/workspace/TCGA_pipeline/data/TCGA_OV_RNA/output_file.tsv'
    DATA_DIR = '/workspace/TCGA_pipeline/data/TCGA_OV_RNA/'
    SAVE_DIR = '/workspace/TCGA_pipeline/data/processed/OV/'
    GENE_LENGTH_FILE_PATH = '/workspace/TCGA_pipeline/data/gencode.v36.geneLength.exon_length'
    CANCER_TYPE = 'OV'

    process_TCGA_RNA_data(BARCODE_MAPPING_PATH,
                          DATA_DIR,
                          GENE_LENGTH_FILE_PATH)

    merge_TCGA_data(DATA_DIR,
                    SAVE_DIR,
                    CANCER_TYPE)


if __name__ == '__main__':
    main()


