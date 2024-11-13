from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class OcelotDataset(Dataset):
    def __init__(self, root_dir, organ, transform=None):
        super(OcelotDataset, self).__init__()
        self.root_dir = Path(root_dir)
        self.organ = organ
        self.transform = transform
        self.patches_dir = Path(root_dir) / 'patches'

        self._input_array = []
        self.y_array = []

        self._read_and_initialise_metadata()
        self._create_input_arrays()

    def __len__(self):
        return len(self._input_array)

    def __getitem__(self, idx):
        x = self.get_input(idx)
        y = self.y_array[idx]

        return x, y

    def _read_and_initialise_metadata(self):
        df = pd.read_csv(self.root_dir / 'metadata.csv', index_col=0,
                         dtype={'file_name': str, 'slide_name': str, 'organ': str, 'tumor': int})
        self.df = df[df['organ'] == self.organ]
        self.df.reset_index(drop=True, inplace=True)

    def _create_input_arrays(self):
        cols = ['file_name', 'x_coord', 'y_coord', 'tumor']
        label_array = []
        for file_name, x, y, tumor in self.df.loc[:, cols].itertuples(index=False, name=None):
            patch_name = f'patch_{file_name}_x_{x}_y_{y}.png'
            path = f'{file_name}/{patch_name}'
            self._input_array.append(path)
            label_array.append(tumor)
        self.y_array = torch.LongTensor(label_array)

    def get_input(self, idx):
        """Returns x, mask_x for a given index(idx)."""
        x = self._read_image(self.patches_dir, idx)
        x = self._transform_image(x, self.transform)

        return x

    def _read_image(self, patches_dir, idx):
        patch_path = patches_dir / self._input_array[idx]
        x = Image.open(patch_path).convert('RGB')

        return x

    @staticmethod
    def _transform_image(image, transform):
        if transform is not None:
            return transform(image)
        return to_tensor(image)
