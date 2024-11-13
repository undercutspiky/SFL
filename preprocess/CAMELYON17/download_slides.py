import os
from multiprocessing import Pool
from pathlib import Path
from zipfile import ZipFile

import wget


def download_training(destination: Path):
    # TODO: Verify the URL. If the url has changed again then find the new one from the links on
    #       https://camelyon17.grand-challenge.org/Data/
    # base_url = 'https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON17/training/'
    base_url = 'https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100439/CAMELYON17/training/'
    directories = ['center_' + str(i) for i in range(5)]

    args = []

    for i, d in enumerate(directories):
        url_dir = base_url + d + '/'
        dst_dir = destination / d
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file_no in range(i*20, (i+1)*20):
            file_name = f'patient_{file_no:03}.zip'
            # Don't download the file if it's already there
            if (dst_dir / file_name).is_file():
                print(f'Not downloading {file_name} as it already exists.')
                continue
            url = url_dir + file_name
            args.append((url, dst_dir))

    with Pool(processes=len(args)) as pool:
        pool.starmap(_download, args)

    # Extract all zip files
    zip_extraction_args = []
    for i, d in enumerate([f'center_{i}' for i in range(5)]):
        dst_dir = destination / d
        for file_name in os.listdir(dst_dir):
            if not file_name.endswith('.zip'):
                continue
            zip_extraction_args.append((dst_dir / file_name, dst_dir))

    with Pool(processes=len(zip_extraction_args)) as pool:
        pool.starmap(_extract_zip_file, zip_extraction_args)


def extract_zip_files(root_dir_path):
    for d in [f'center_{i}' for i in range(5)]:
        dst_dir = Path(root_dir_path) / d
        for file_name in os.listdir(dst_dir):
            if not file_name.endswith('.zip'):
                continue
            print(f'Working on {file_name}')
            with ZipFile(dst_dir / file_name, 'r') as zf:
                zf.extractall(dst_dir)
            os.remove(dst_dir / file_name)


def _download(url: str, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url} at {str(dst)}:')
    wget.download(url, out=str(dst))


def _extract_zip_file(zip_file_path: Path, dst: Path):
    print(f'Extracting {zip_file_path}')
    with ZipFile(zip_file_path, 'r') as zip_obj:
        zip_obj.extractall(dst)


if __name__ == '__main__':
    # TODO: Modify the root directory in the path below
    download_training(Path('/data/camelyon17/training/'))
