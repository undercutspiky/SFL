# Code adapted from https://github.com/liucong3/camelyon17
# and https://github.com/cv-lee/Camelyon17

import argparse
import json
import os
from pathlib import Path

import cv2
import pandas as pd

PATCH_LEVEL = 0
MASK_LEVEL = 2
CENTER_SIZE = 90
NORMAL_THRESHOLD = 1
TUMOUR_THRESHOLD = 0


def _make_masks(slide_path, mask_path):
    """
    Return a slide with annotated tumor, normal, and tissue masks using an Otsu threshold
    """
    print(f'_make_masks on {os.path.basename(slide_path)}')

    slide_map = cv2.imread(slide_path)
    slide_map = cv2.cvtColor(slide_map, cv2.COLOR_BGR2RGB)

    slide_labels = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w, _ = slide_map.shape
    slide_labels = cv2.resize(slide_labels, (w, h), interpolation=cv2.INTER_CUBIC)
    tumor_mask = slide_labels == 2
    normal_mask = slide_labels == 1

    return slide_map, tumor_mask, normal_mask


def _record_patches(slide_map, tumor_mask, normal_mask):
    """
    Extract all tumor and non-tumor patches from a slide, using the given masks.
    """

    # Extract normal, tumor patches using normal, tumor mask
    height, width = slide_map.shape[0] // CENTER_SIZE, slide_map.shape[1] // CENTER_SIZE

    print(f'_record_patches(w={width},h={height})')
    margin = 5  # 3
    mask_max = 1

    width_mask_step = CENTER_SIZE
    height_mask_step = CENTER_SIZE

    patch_list = []
    pos_count, neg_count = 0, 0

    # These mark the coordinates of the central region of the patch
    for i in range(margin, width - margin):
        for j in range(margin, height - margin):

            mask_i_start = round(width_mask_step * i)
            mask_i_end = round(width_mask_step * (i + 1))
            mask_j_start = round(height_mask_step * j)
            mask_j_end = round(height_mask_step * (j + 1))

            # Compute masks only over central region
            tumor_mask_avg = tumor_mask[
                             mask_j_start: mask_j_end,
                             mask_i_start: mask_i_end].mean()
            normal_mask_avg = normal_mask[
                              mask_j_start: mask_j_end,
                              mask_i_start: mask_i_end].mean()
            # Calculate tissue avg to avoid having non-tissue slide background as normal tissue
            tissue_avg = slide_map[mask_j_start: mask_j_end, mask_i_start: mask_i_end].mean()

            tumor_area_ratio = tumor_mask_avg / mask_max
            normal_area_ratio = normal_mask_avg / mask_max

            # Extract patch coordinates
            # Coords correspond to the center only, not the entire patch
            if tumor_area_ratio > TUMOUR_THRESHOLD:
                patch_list.append((CENTER_SIZE * i, CENTER_SIZE * j, 1))
                pos_count += 1

            elif normal_area_ratio >= NORMAL_THRESHOLD and tissue_avg < 243:  # threshold tissue_avg to avoid white bg
                patch_list.append((CENTER_SIZE * i, CENTER_SIZE * j, 0))
                neg_count += 1

    print(f'patch_list length = {len(patch_list)}, pos_count={pos_count}, neg_count={neg_count}')

    df = pd.DataFrame(patch_list, columns=['x_coord', 'y_coord', 'tumor'])
    return df


def generate_file(mask_path, slide_path):
    slide_map, tumor_mask, normal_mask = _make_masks(slide_path, mask_path)
    df = _record_patches(slide_map, tumor_mask, normal_mask)

    return df


def generate_files(ocelot_root, output_root):
    ocelot_root = Path(ocelot_root)
    with open(ocelot_root / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    aggregate_df = pd.DataFrame(
        columns=[
            'slide_name',
            'file_name',
            'organ',
            'x_coord',
            'y_coord',
            'tumor'
        ])

    for file_name, data in metadata['sample_pairs'].items():
        file_path = ocelot_root / 'images' / f'{file_name}.png'
        if not file_path.exists():
            print(f'Skipping {file_name} as it does not exist hopefully cuz mpp (in the next sentence) is > 0.26.',
                  f'MPP_X = {data["mpp_x"]}')
            continue
        mask_path = ocelot_root / 'annotations' / f'{file_name}.png'
        df = generate_file(str(mask_path), str(file_path))
        df['slide_name'] = data['slide_name']
        df['file_name'] = file_name
        df['organ'] = data['organ']
        aggregate_df = pd.concat([aggregate_df, df])

    aggregate_df = aggregate_df.reset_index(drop=True)
    aggregate_df.to_csv(os.path.join(output_root, 'all_patch_coords.csv'))
    return aggregate_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_root', default='/data/ocelot2023_v1.0.1/')
    parser.add_argument('--output_root', default='/data/ocelot2023_v1.0.1/')
    _args = parser.parse_args()

    generate_files(ocelot_root=_args.slide_root, output_root=_args.output_root)
