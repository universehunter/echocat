from random import shuffle
from PIL import Image
import copy
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import torch

from core.datasets.compose import Compose


class Mydataset(Dataset):
    def __init__(self, gt_labels, cfg):
        self.gt_labels = gt_labels
        self.cfg = cfg
        self.pipeline = Compose(self.cfg)
        self.data_infos = self.load_annotations()

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # Get raw data info
        data_info = self.data_infos[index]

        # Process through pipeline using deepcopy
        results = self.pipeline(copy.deepcopy(data_info))

        # Fix: Safely get gt_label.
        # If pipeline drops it (common in validation), retrieve from raw data_info.
        if 'gt_label' in results:
            label = results['gt_label']
        else:
            label = data_info['gt_label']

        # Ensure label is int
        if isinstance(label, np.ndarray):
            label = label.item()

        # Safely get filename
        filename = results.get('filename', data_info['img_info']['filename'])

        return results['img'], int(label), filename

    def load_annotations(self):
        """Load image paths and gt_labels."""
        if not self.gt_labels:
            raise TypeError('ann_file is None or empty')

        data_infos = []

        for x in self.gt_labels:
            line = x.strip()
            if not line:
                continue

            # Fix: Use rsplit to handle paths with spaces correctly
            # This splits from the right side once, separating path and label
            parts = line.rsplit(' ', 1)

            if len(parts) < 2:
                continue

            filename, gt_label = parts

            info = {'img_prefix': None}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)

        return data_infos


def collate(batches):
    images, gts, image_path = tuple(zip(*batches))
    images = torch.stack(images, dim=0)
    gts = torch.as_tensor(gts)

    return images, gts, image_path