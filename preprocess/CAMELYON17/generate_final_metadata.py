import argparse
import os

import numpy as np
import pandas as pd


def generate_final_metadata(output_root):
    df = pd.read_csv(os.path.join(output_root, 'all_patch_coords.csv'),
                     index_col=0, dtype={'patient': 'str', 'tumor': 'int'})
    # Assign slide numbers to patients + nodes
    patient_node_list = list(set(df[['patient', 'node']].itertuples(index=False, name=None)))
    patient_node_list.sort()
    patient_node_to_slide_map = {}
    for idx, (patient, node) in enumerate(patient_node_list):
        patient_node_to_slide_map[(patient, node)] = idx

    for (patient, node), slide_idx in patient_node_to_slide_map.items():
        mask = (df['patient'] == patient) & (df['node'] == node)
        df.loc[mask, 'slide'] = slide_idx
    df['slide'] = df['slide'].astype('int')

    # The raw data has the following assignments:
    # Center 0: patients 0 to 19
    # Center 1: patients 20 to 39
    # Center 2: patients 40 to 59
    # Center 3: patients 60 to 79
    # Center 4: patients 80 to 99
    num_centers = 5
    patients_per_center = 20
    df['center'] = df['patient'].astype('int') // patients_per_center

    for k in range(num_centers):
        print(f"center {k}: "
              f"{np.sum((df['center'] == k) & (df['tumor'] == 0)):6d} non-tumor, "
              f"{np.sum((df['center'] == k) & (df['tumor'] == 1)):6d} tumor")

    for center, slide in set(df[['center', 'slide']].itertuples(index=False, name=None)):
        assert center == slide // 10

    indices_to_keep = get_indices_to_keep(df)
    # Within each center, keep same number of normal patches as tumor patches
    df_to_keep = get_balanced_center_df(df, indices_to_keep, num_centers)

    for _, center_df in df_to_keep.groupby('center'):
        print('for center', center_df.iloc[0]['center'], 'of length', len(center_df))
        count = (len(center_df[center_df['tumor'] == 0]), len(center_df[center_df['tumor'] == 1]))
        for _, p_g in center_df.groupby('patient'):
            print('\t', p_g.iloc[0]['patient'], [(i.iloc[0]['tumor'], len(i)) for _, i in p_g.groupby('tumor')])
            percentages = [(i.iloc[0]["tumor"], str(round(len(i)*100/count[idx], 2)) + '%')
                           for idx, i in p_g.groupby('tumor')]
            print(f'\t\t {percentages}')

    val_patient_ids = {
        0: ['010', '015'],
        1: ['024', '034'],
        2: ['044', '052'],
        3: ['061', '062', '075'],
        4: ['092', '099']
    }
    split_dict = {'train': 0, 'val': 1}
    df_to_keep['split'] = split_dict['train']

    val_patients = [x for v in val_patient_ids.values() for x in v]
    val_indices = list(df_to_keep[df_to_keep['patient'].isin(val_patients)].index)
    df_to_keep.loc[val_indices, 'split'] = split_dict['val']

    print('Statistics by center:')
    for center in range(num_centers):
        tumor_mask = df_to_keep['tumor'] == 1
        center_mask = df_to_keep['center'] == center
        num_tumor = np.sum(center_mask & tumor_mask)
        num_non_tumor = np.sum(center_mask & ~tumor_mask)

        print(f'Center {center}')
        print(f'\t{num_tumor} / {num_tumor + num_non_tumor} ({num_tumor / (num_tumor + num_non_tumor) * 100:.1f}%) tumour')

    df_to_keep.to_csv(os.path.join(output_root, 'metadata.csv'))


def get_balanced_center_df(df, indices_to_keep, num_centers):
    tumor_keep_mask = np.zeros(len(df))
    tumor_keep_mask[df.index[indices_to_keep]] = 1
    tumor_mask = df['tumor'] == 1
    for center in range(num_centers):
        print(f'Center {center}:')
        center_mask = df['center'] == center
        num_tumor = np.sum(center_mask & tumor_keep_mask)
        print(f'\tNum tumor: {num_tumor}')

        num_non_tumor = np.sum(center_mask & ~tumor_mask)
        center_indices_without_tumor = list(df.index[center_mask & ~tumor_mask])
        indices_to_keep += list(np.random.choice(
            center_indices_without_tumor,
            size=min(num_tumor, num_non_tumor),
            replace=False))

        print(f'\tNum non-tumor: {min(num_tumor, num_non_tumor)} out of {num_non_tumor} '
              f'({min(num_tumor, num_non_tumor) / num_non_tumor * 100:.1f}%)')

        df_to_keep = df.loc[indices_to_keep, :].copy().reset_index(drop=True)
    return df_to_keep


def get_indices_to_keep(df):
    indices_to_keep = []
    np.random.seed(0)
    tumor_mask = df['tumor'] == 1

    for slide in set(df['slide']):
        slide_mask = (df['slide'] == slide)
        num_tumor = np.sum(slide_mask & tumor_mask)
        num_non_tumor = np.sum(slide_mask & ~tumor_mask)
        slide_indices_with_tumor = list(df.index[slide_mask & tumor_mask])
        indices_to_keep += list(np.random.choice(
            slide_indices_with_tumor,
            size=min(num_tumor, num_non_tumor),
            replace=False))

    return indices_to_keep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', default='/data/camelyon17')
    args = parser.parse_args()

    generate_final_metadata(args.output_root)
