import os
import glob
import numpy as np
from tqdm import tqdm
from typing import Optional

from skimage.io import imread
from tifffile import imread as tiff_imread

import torch

from lsm.utils.io_util import glob_imgs


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: Optional[str] = "Training"):

        assert os.path.exists(data_dir), f"Data directory does not exist"

        self.root_dir = data_dir

        image_dir = f"{self.root_dir}/{split}/images/"
        image_paths = sorted(glob_imgs(image_dir))
        label_dir = f"{self.root_dir}/{split}/labels"

        self.images_all = []
        self.labels_all = []

        # not all png files have corresponding masks for some reason
        for image in tqdm(image_paths, f"Loading {split} images..."):
            try:
                img_name = image.split("/")[-1]
                label_name = img_name[: img_name.rfind(".")] + "_label.tiff"
                label_path = os.path.join(label_dir, label_name)

                assert os.path.exists(label_path)

                rgb = torch.from_numpy(imread(image))
                label = torch.from_numpy(tiff_imread(label_path).astype(np.int16))

                self.images_all.append(rgb)
                self.labels_all.append(label)

            except:
                continue

        self.n_images = len(self.images_all)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):

        sample = {"rgb": self.images_all[idx]}
        ground_truth = {"mask": self.labels_all[idx]}

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)


if __name__ == "__main__":
    # testing dataloader
    data_dir = "/om2/user/ckapoor/lsm-data/NeurIPS22-CellSeg"
    data = ImageDataset(data_dir=data_dir, split="Training")
    _, sample, label = data[0]
    rgb = sample["rgb"]
    mask = label["mask"]

    print(f"rgb shape: {rgb.shape}\t mask shape: {mask.shape}")
    print(f"len: {len(data)}")
