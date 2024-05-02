import os
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from natsort import natsorted

from cellpose import io, transforms

import torch

from lsm.utils.io_util import glob_imgs


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: Optional[str] = "Testing"):

        assert os.path.exists(data_dir), f"Data directory does not exist"

        self.data_dir = Path(data_dir)
        fall = natsorted(glob((self.data_dir / "images" / "*").as_posix()))
        img_files = [f for f in fall if "_masks" not in f and "_flows" not in f]

        # self.imgs = [torch.from_numpy(io.imread(f)) for f in img_files]
        self.imgs = [io.imread(f) for f in img_files]

        self.imgs_norm = []
        for img in self.imgs:
            if img.ndim == 2:
                img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
            img = transforms.normalize_img(img, axis=-1)
            self.imgs_norm.append(img.transpose(2, 0, 1))

        self.n_images = len(self.imgs)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):

        sample = {"norm_rgb": self.imgs_norm[idx].astype(np.float32)}
        ground_truth = {"orig_rgb": self.imgs[idx].astype(np.float32)}

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
    data_dir = "/om2/user/ckapoor/lsm-data/NeurIPS22-CellSeg-2/Testing/Hidden"
    data = ImageDataset(data_dir=data_dir, split="Training")
    _, sample, label = data[0]
    rgb = sample["norm_rgb"]
    mask = label["orig_rgb"]

    import cv2 as cv

    for i, _ in enumerate(data):
        _, norm, _ = data[i]
        rgb = norm["norm_rgb"]
        rgb = rgb.transpose(1, 2, 0)
        print(f"rgb: {rgb.shape}")
        cv.imwrite(f"{i}_norm_dl.png", rgb)

    print(f"rgb shape: {rgb.shape}\t mask shape: {mask.shape}")
    print(f"len: {len(data)}")
