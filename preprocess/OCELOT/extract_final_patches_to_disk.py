import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm

from preprocess.OCELOT.generate_all_patch_coords import CENTER_SIZE


def write_patch_images_from_df(slide_root, output_root):
    df = pd.read_csv(os.path.join(slide_root, 'metadata.csv'), index_col=0,
                     dtype={'file_name': str, 'slide_name': str, 'organ': str, 'tumor': int})

    center_size = CENTER_SIZE
    patch_size = center_size * 3

    for name, group in df.groupby('file_name'):
        print(f'Working on {name}')
        slide = cv2.imread(os.path.join(slide_root, 'images', f'{name}.png'))

        for idx in tqdm(group.index):
            orig_x = df.loc[idx, 'x_coord']
            orig_y = df.loc[idx, 'y_coord']

            patch_folder = os.path.join(output_root, 'patches', name)
            patch_path = os.path.join(patch_folder, f'patch_{name}_x_{orig_x}_y_{orig_y}.png')

            os.makedirs(patch_folder, exist_ok=True)
            if os.path.isfile(patch_path):
                continue

            # Coords are at patch_level
            # First shift coords to top left corner of the entire patch
            x = orig_x - center_size
            y = orig_y - center_size

            patch = slide[y: y + patch_size, x: x + patch_size]
            cv2.imwrite(patch_path, patch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_root', default='/data/ocelot2023_v1.0.1/')
    parser.add_argument('--output_root', default='/data/ocelot2023_v1.0.1/')
    args = parser.parse_args()

    write_patch_images_from_df(slide_root=args.slide_root, output_root=args.output_root)
