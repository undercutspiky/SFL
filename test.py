import argparse
import os
import re
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet50

from train import get_dataloader

cuda_device = torch.device('cuda', 0)


def main(ckpts_root_path: Path, data_dir):
    models = [f'{ckpts_root_path}/{i}' for i in os.listdir(ckpts_root_path) if i.endswith('.pth')]

    print(f'\n\n{"=" * 80}\nEvaluating models on CAMELYON17.\n{"=" * 80}\n\n')
    print(f'Gotta run evals on {len(models)} models!')
    for m in models:
        run_eval(m, data_dir)


def run_eval(checkpoint_path, data_dir):
    config = {'DISTANCE_START_EPOCH': 1, 'TOTAL_EPOCHS': 0}
    print(f'Evaluating {checkpoint_path}', flush=True)
    hospital = int(re.search(r"h(\d+)", Path(checkpoint_path).stem).group(1))
    model = resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.to(cuda_device)
    model.eval()

    h0_dataloader = get_dataloader(data_dir, hospital=0, split=('val' if hospital == 0 else 'all'), batch_size=128,
                                   config=config, num_workers=9)
    h1_dataloader = get_dataloader(data_dir, hospital=1, split=('val' if hospital == 1 else 'all'), batch_size=128,
                                   config=config, num_workers=9)
    h2_dataloader = get_dataloader(data_dir, hospital=2, split=('val' if hospital == 2 else 'all'), batch_size=128,
                                   config=config, num_workers=9)
    h3_dataloader = get_dataloader(data_dir, hospital=3, split=('val' if hospital == 3 else 'all'), batch_size=128,
                                   config=config, num_workers=9)
    h4_dataloader = get_dataloader(data_dir, hospital=4, split=('val' if hospital == 4 else 'all'), batch_size=128,
                                   config=config, num_workers=9)

    print(f'Evaluating model on h0 on split {h0_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h0_dataloader)
    print(f'Evaluating on h0 took {timedelta(seconds=time.time() - start_time)}\n{"="*70}\n', flush=True)

    print(f'Evaluating model on h1 on split {h1_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h1_dataloader)
    print(f'Evaluating on h1 took {timedelta(seconds=time.time() - start_time)}\n{"=" * 70}\n', flush=True)

    print(f'Evaluating model on h2 on split {h2_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h2_dataloader)
    print(f'Evaluating on h2 took {timedelta(seconds=time.time() - start_time)}\n{"=" * 70}\n', flush=True)

    print(f'Evaluating model on h3 on split {h3_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h3_dataloader)
    print(f'Evaluating on h3 took {timedelta(seconds=time.time() - start_time)}\n{"=" * 70}\n', flush=True)

    print(f'Evaluating model on h4 on split {h4_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h4_dataloader)
    print(f'Evaluating on h4 took {timedelta(seconds=time.time() - start_time)}\n{"=" * 70}\n', flush=True)


def evaluate_model(model, dataloader):
    confusion_mat = 0
    with torch.no_grad():
        for data in dataloader:
            (x, _), labels, _ = data
            x = x.to(cuda_device, memory_format=torch.channels_last)
            labels = labels.unsqueeze(1).float()
            outputs = torch.sigmoid(model(x))
            preds = outputs > 0.5
            confusion_mat += confusion_matrix(labels, preds.cpu(), labels=[0, 1])

        acc = np.sum(np.diag(confusion_mat) / np.sum(confusion_mat))

        print('Accuracy(mean): %f %%' % (100 * acc))
        print(pd.DataFrame(confusion_mat, index=['NoTumour', 'Tumour'], columns=['NoTumour', 'Tumour']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval on all baseline models or other models.')
    parser.add_argument('--ckpts_path', type=Path, help='Path to the checkpoint files.')
    parser.add_argument('--data_dir', type=str, help='Path to the CAMELYON17 data directory.')
    args = parser.parse_args()

    main(args.ckpts_path, args.data_dir)
