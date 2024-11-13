import argparse
import os

import numpy as np
import pandas as pd


def generate_final_metadata(output_root):
    df = pd.read_csv(os.path.join(output_root, 'all_patch_coords.csv'),
                     index_col=0, dtype={'file_name': str, 'slide_name': str, 'organ': str, 'tumor': int})
    indices_to_keep = []
    np.random.seed(0)
    tumor_mask = df['tumor'] == 1
    for slide in set(df['slide_name']):
        slide_mask = (df['slide_name'] == slide)
        num_tumor = np.sum(slide_mask & tumor_mask)
        num_non_tumor = np.sum(slide_mask & ~tumor_mask)
        # Remove some tumour patches if their number exceeds the non-tumour patches
        # Regret: I don't know why I didn't do it the other way round but since this
        # is an external test set, I didn't want to change the dataset once it had
        # been created and tested on
        slide_indices_with_tumor = list(df.index[slide_mask & tumor_mask])
        indices_to_keep += list(np.random.choice(
            slide_indices_with_tumor,
            size=min(num_tumor, num_non_tumor),
            replace=False))

    tumor_keep_mask = np.zeros(len(df))
    tumor_keep_mask[df.index[indices_to_keep]] = 1

    # Within each organ, balance the number of normal patches and tumor patches
    for organ in set(df['organ']):
        print(f'Organ {organ}:')
        organ_mask = df['organ'] == organ
        num_tumor = np.sum(organ_mask & tumor_keep_mask)
        print(f'  Num tumor: {num_tumor}')

        num_non_tumor = np.sum(organ_mask & ~tumor_mask)
        organ_indices_without_tumor = list(df.index[organ_mask & ~tumor_mask])
        indices_to_keep += list(np.random.choice(
            organ_indices_without_tumor,
            size=min(num_tumor, num_non_tumor),
            replace=False))

        print(f'  Num non-tumor: {min(num_tumor, num_non_tumor)} out of {num_non_tumor} '
              f'({min(num_tumor, num_non_tumor) / num_non_tumor * 100:.1f}%)')

        df_to_keep = df.loc[indices_to_keep, :].copy().reset_index(drop=True)

    df_to_keep.to_csv(os.path.join(output_root, 'metadata.csv'))
    print(df_to_keep.groupby(['organ', 'tumor']).size())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', default='/data/ocelot2023_v1.0.1/')
    args = parser.parse_args()

    generate_final_metadata(args.output_root)
