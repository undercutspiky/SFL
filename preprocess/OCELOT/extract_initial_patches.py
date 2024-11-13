import json
from pathlib import Path

import openslide


def read_and_store_patches(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Modify the path below so that it points to the correct location of metadata.json
    metadata = read_metadata('metadata.json')
    for key, val in metadata.items():
        if val['mpp_x'] > 0.26 or val['mpp_y'] > 0.26:
            print(f'Skipping {key} cuz mpp_x={val["mpp_x"]} & mpp_y={val["mpp_y"]}')
            continue
        print(f'Working on {key}')
        dst_path = dst_dir / f'{key}.png'
        x = val['tissue']['x_start']
        y = val['tissue']['y_start']
        patch_size_x = val['tissue']['x_end'] - x
        patch_size_y = val['tissue']['y_end'] - y

        slide = openslide.OpenSlide(src_dir / 'wsis' / f'{val["slide_name"]}.svs')
        patch = slide.read_region((x, y), 0, (patch_size_x, patch_size_y))
        patch.save(dst_path)


def read_metadata(metadata_file_path):
    with open(metadata_file_path, 'r') as f:
        data = json.load(f)
    return data['sample_pairs']


if __name__ == '__main__':
    # TODO: modify the paths below
    read_and_store_patches('/data/ocelot2023_v1.0.1', '/data/ocelot2023_v1.0.1/images')
