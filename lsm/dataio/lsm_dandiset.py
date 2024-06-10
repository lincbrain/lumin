import numpy as np
from tqdm import tqdm
from typing import Optional, List

import torch

from cellpose import transforms

import dask
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


class ImageDataset(torch.utils.data.Dataset):
    """
    (lazy) load a zarr file at a specific scale
    """

    def __init__(
        self,
        url: str,
        vol_lim: List[int],
        voxel_shapes: List[int],
        data_dir: Optional[str] = None,
        scale: Optional[int] = 0,
    ):

        self.url = url
        self.vol_lim = vol_lim
        self.voxel_shapes = voxel_shapes
        self.scale = scale

    def __len__(self):
        return max(self.voxel_shapes)

    def __getitem__(self, idx):
        vol = self.read_vol()
        subvoxel = self.create_subvol(voxel=vol)
        sample = {"orig_vol": subvoxel}
        return idx, sample

    def create_subvol(self, voxel):
        # for the most part, i didn't need the entire vol
        # which is why i take (atmost) a 512^3 subvoxel for analysis
        # this could very well be generalized to use the entire volumes
        # dimension instead, by setting appropriate limits in the config
        subvol = voxel[
            self.vol_lim[0] : self.vol_lim[0] + self.voxel_shapes[0],
            self.vol_lim[1] : self.vol_lim[1] + self.voxel_shapes[1],
            self.vol_lim[2] : self.vol_lim[2] + self.voxel_shapes[2],
        ]
        return subvol

    def read_vol(self):
        # stream volume of a specific scale from a url
        reader = Reader(parse_url(self.url))
        dask_vol = list(reader())[0].data
        dask_vol_scale = dask_vol[self.scale][0]

        # order of axes: (c, z, y, x) -> (z, y, x, c)
        return np.transpose(dask_vol_scale, (1, 2, 3, 0))

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
    scale = 0
    vol_scale = dask_data[scale]
    print(f"vs: {vol_scale}")
