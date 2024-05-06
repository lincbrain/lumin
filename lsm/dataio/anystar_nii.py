import os
import numpy as np
from glob import glob
import nibabel as nib
from typing import Optional

import torch


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: Optional[str] = "Train"):

        assert os.path.exists(data_dir), f"Data directory does not exist"

        self.data_dir = data_dir
        img_files = sorted(glob(self.data_dir + "/*.nii.gz"))

        self.imgs = [nib.load(f).get_fdata() for f in img_files]
        self.imgs_norm = self.imgs

        self.n_images = len(self.imgs)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # misnomer here; all "norm" images are same as GT
        sample = {"norm_rgb": self.imgs_norm[idx]}
        ground_truth = {"orig_rgb": self.imgs[idx]}

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
    data_dir = "/om2/user/ckapoor/lsm-segmentation/anystar-data/NucMM-Z"
    data = ImageDataset(data_dir=data_dir)
    _, sample, label = data[0]
    rgb = sample["norm_rgb"]
    gt = label["orig_rgb"]
    import cv2 as cv

    # save the 40th (arbitrary) slice for sanity check
    cv.imwrite(f"norm_rgb.png", rgb[..., 40])
    cv.imwrite(f"orig_rgb.png", gt[..., 40])
    print(f"len: {len(data)}")
