import numpy as np
from tqdm import tqdm
from typing import Optional

import torch

from cellpose import transforms

import dask
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, url: str, scale: Optional[int] = 0):
        """
        remotely stream a zarr array from a specified url
        """
        self.scale = scale
        reader = Reader(parse_url(url))

        self.dask_data = list(reader())[0].data

        # trigger dask array conversion to numpy
        img = self.dask_to_np(group=self.dask_data)
        self.img_stack = img
        print(f"im stack: {self.img_stack.shape}")

        # normalize volume
        self.imgs_norm = []
        for tile in tqdm(range(img.shape[-1]), desc="Normalizing volume..."):
            single_slice = img[..., tile]
            if single_slice.ndim == 2:
                single_slice = np.tile(single_slice[:, :, np.newaxis], (1, 1, 3))
            single_slice = transforms.normalize_img(single_slice, axis=-1)
            self.imgs_norm.append(single_slice.transpose(2, 0, 1))

        self.imgs_norm = np.stack(self.imgs_norm, axis=0)  # [Z, C, X, Y]

        self.n_slices = self.imgs_norm.shape[0]

    def __len__(self):
        return self.n_slices

    def dask_to_np(self, group: dask.array):
        # note: array is loaded in format [z, y, x]
        group = group[self.scale]
        # group -> [T, C, X, Y, Z]

        return group[0][0].compute().transpose(2, 1, 0)
        # return group.reshape(group.shape[2:]).compute()

    def __getitem__(self, idx):

        sample = {"norm_rgb": self.imgs_norm[idx].astype(np.float32)}
        ground_truth = {
            "orig_rgb": self.img_stack[..., idx].astype(np.float32)
        }  # get corresponding z-slice

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
    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
    reader = Reader(parse_url(url))
    dask_data = list(reader())[0].data
    scale = 2
    vol_scale = dask_data[scale]
    vol = vol_scale[0][0]  # (z, y, x)

    from tifffile import imsave

    print(f"shape: {vol.shape}")
    print(f"dtype: {vol[100].dtype}")
    # color_map = np.random.randint(0, 256, (25890, 3), dtype=np.uint8)
    # color_map[0] = [0, 0, 0]
    # sl = color_map[sl]
    # print(f"uq: {np.unique(sl)}")
    imsave(f"test_sl100.tiff", vol[100])

    # d = ImageDataset(url=url, scale=3)
    # _, model_ip, gt = d[500]

    # from cellpose import models

    # model = models.Cellpose(gpu=True, nchan=3, model_type="neurips_cellpose_default")
    # channels = None
    # normalize = False
    # diams = None

    # img_slice = model_ip["norm_rgb"]
    # print(f"sl uq: {np.unique(img_slice)}")
    # out = model.eval(
    #    img_slice,
    #    diameter=diams,
    #    channels=channels,
    #    normalize=normalize,
    #    do_3D=True,
    #    channel_axis=1,
    #    tile_overlap=0.6,
    #    augment=True,
    # )

    # mask = out[0]

    # gt_img = gt["orig_rgb"]
    # import cv2 as cv

    # print(f"m: {mask.shape}")
    # color_map = np.random.randint(0, 256, (500, 3), dtype=np.uint8)
    # color_map[0] = [0, 0, 0]
    # seg_mask = color_map[mask[5]]
    # cv.imwrite(f"mask_lsm.png", seg_mask[5])
    # cv.imwrite(f"rgb_lsm.png", gt_img)
