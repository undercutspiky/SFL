import argparse
import os

import numpy as np
import pandas as pd


def generate_final_metadata(output_root):
    df = pd.read_csv(os.path.join(output_root, 'all_patch_coords.csv'),
                     index_col=0, dtype={'file_name': str, 'tumor': int})
    # Assign slide numbers to patients + nodes
    patient_node_list = list(set(df[['file_name']].itertuples(index=False, name=None)))
    patient_node_list.sort()
    patient_node_to_slide_map = {}
    for idx, file_name in enumerate(patient_node_list):
        patient_node_to_slide_map[file_name] = idx

    for file_name, slide_idx in patient_node_to_slide_map.items():
        mask = (df['file_name'] == file_name[0])
        df.loc[mask, 'slide'] = slide_idx
    df['slide'] = df['slide'].astype('int')

    # Extract the same number of tumour and non-tumour patches from each WSI
    indices_to_keep = []
    np.random.seed(0)
    tumor_mask = df['tumor'] == 1
    for slide in set(df['slide']):
        slide_mask = (df['slide'] == slide)
        num_tumor = np.sum(slide_mask & tumor_mask)
        num_non_tumor = np.sum(slide_mask & ~tumor_mask)
        # Number of tumour patches to retain = min(num_tumor, num_non_tumor)
        slide_indices_with_tumor = list(df.index[slide_mask & tumor_mask])
        indices_to_keep += list(np.random.choice(
            slide_indices_with_tumor,
            size=min(num_tumor, num_non_tumor),
            replace=False))
        # Number of non-tumour patches to retain = min(num_tumor, num_non_tumor)
        slide_indices_without_tumor = list(df.index[slide_mask & ~tumor_mask])
        indices_to_keep += list(np.random.choice(
            slide_indices_without_tumor,
            size=min(num_tumor, num_non_tumor),
            replace=False))

    df_to_keep = df.loc[indices_to_keep, :].copy().reset_index(drop=True)
    df_to_keep.to_csv(os.path.join(output_root, 'metadata.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', default='/data/BCSS/')
    args = parser.parse_args()
    generate_final_metadata(args.output_root)
