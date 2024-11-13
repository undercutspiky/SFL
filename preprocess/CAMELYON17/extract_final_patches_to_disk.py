"""
This code assumes that there are nuclear segmentation masks available for each WSI.
This is because the work on the project after the NeurIPS paper used HoverFast for
nuclear segmentation. And it was just easier to extract patches for both images
and their corresponding mask patches at the same time.
Please remove the code related to mask patch extraction if needed.
"""
import argparse
import gzip
import json
import os
from pathlib import Path

import openslide
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

from preprocess.CAMELYON17.generate_all_patch_coords import PATCH_LEVEL, CENTER_SIZE


def write_patch_images_from_df(slide_root, output_root):
    read_df = pd.read_csv(
        os.path.join(output_root, 'metadata.csv'),
        index_col=0,
        dtype={'patient': str})

    patch_level = PATCH_LEVEL
    center_size = CENTER_SIZE
    patch_size = center_size * 3

    for (patient, node), df in read_df.groupby(['patient', 'node']):
        print(f'Working on patient_{patient}_node_{node}.tif')
        center_dir_name = f'center_{int(patient) // 20}'
        slide_path = os.path.join(slide_root, 'training', center_dir_name, f'patient_{patient}_node_{node}.tif')
        slide = openslide.OpenSlide(slide_path)
        width, height = slide.dimensions
        # Read the nuclear segmentation file generated via HoverFast (https://github.com/choosehappy/HoverFast)
        hoverfast_file_path = Path(slide_root) / 'nuclear_segmentation_masks' / f'patient_{patient}_node_{node}.json.gz'
        mask_exists = os.path.exists(hoverfast_file_path)
        if mask_exists:
            mask = get_nuclear_segmentation(hoverfast_file_path, width, height)

        for idx in tqdm(df.index):
            orig_x = df.loc[idx, 'x_coord']
            orig_y = df.loc[idx, 'y_coord']

            patch_folder = os.path.join(output_root, 'tissue', 'patches', f'patient_{patient}_node_{node}')
            patch_path = os.path.join(patch_folder, f'patch_patient_{patient}_node_{node}_x_{orig_x}_y_{orig_y}.png')
            os.makedirs(patch_folder, exist_ok=True)

            if mask_exists:
                mask_folder = os.path.join(output_root, 'hoverfast', 'patches', f'patient_{patient}_node_{node}')
                mask_path = os.path.join(mask_folder, f'patch_patient_{patient}_node_{node}_x_{orig_x}_y_{orig_y}.png')
                os.makedirs(mask_folder, exist_ok=True)

            # Move on to the next patch if this patch has already been extracted
            if os.path.isfile(patch_path):
                continue

            # Coords are at patch_level
            # First shift coords to top left corner of the entire patch
            x = orig_x - center_size
            y = orig_y - center_size
            # Then match to level 0 coords so we can use read_region
            x = int(round(x * slide.level_dimensions[0][0] / slide.level_dimensions[patch_level][0]))
            y = int(round(y * slide.level_dimensions[0][1] / slide.level_dimensions[patch_level][1]))

            patch = slide.read_region((x, y), patch_level, (patch_size, patch_size))
            patch.save(patch_path)

            if mask_exists:
                mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))
                mask_patch.save(mask_path)


def get_nuclear_segmentation(hoverfast_file_path, width, height):
    with gzip.open(hoverfast_file_path, 'r') as f:
        json_bytes = f.read()
    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)

    print('Creating blank mask image.', end=' ', flush=True)
    mask = Image.new("RGB", (width, height), (0, 0, 0))
    print('Filling image with polygons', end=' ', flush=True)
    for i, n in enumerate(data):
        if len(n['geometry']['coordinates']) != 1:
            print(i, end=' ')
        poly = [tuple(i) for i in n['geometry']['coordinates'][0]]
        ImageDraw.Draw(mask).polygon(poly, outline=(255, 255, 255), fill=(255, 255, 255))
    print('Done creating nuclear mask image.', flush=True)
    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_root', default='/data/camelyon17/')
    parser.add_argument('--output_root', default='/data/camelyon17/')
    args = parser.parse_args()

    write_patch_images_from_df(slide_root=args.slide_root, output_root=args.output_root)
