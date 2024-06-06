import os
import operator
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple

from stardist.models import StarDist3D


if __name__ == "__main__":
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
    reader = Reader(parse_url(url))
    dask_data = list(reader())[0].data
    scale = 0
    vol_scale = dask_data[scale][0]
    print(f"vs shape: {vol_scale.shape}")

    vol_scale = np.transpose(vol_scale, (1, 2, 3, 0))  # (c, z, y, x) -> (z, y, x, c)

    chunk_size = 64
    voxel = vol_scale[
        1000 : 1000 + chunk_size, 650 : 650 + chunk_size, 3500 : 3500 + chunk_size
    ]

    from tifffile import imsave

    debug_dir = f"./chunk64_stardist_debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # model_name = "gaussian_steerable_run"
    model_name = "anystar-mix"
    model_folder = "models"
    model = StarDist3D(None, name=model_name, basedir=model_folder)
    model.load_weights(name="weights_best.h5")
    model.trainable = False
    model.keras_model.trainable = False

    # normalize voxel

    x = voxel.compute()
    upper = np.percentile(x, 99.9)
    x = np.clip(x, 0, upper)
    x = (x - x.min()) / (x.max() - x.min())
    x = x[..., 0]

    seg, _ = model.predict_instances(
        x,
        prob_thresh=0.67,
        nms_thresh=0.3,
        n_tiles=None,
        scale=[1.0, 1.0, 1.0],
    )

    seg = np.asarray(seg)
    from tifffile import imwrite

    imwrite(f"./{debug_dir}/pred_anystar_seg.tiff", seg)
    imwrite(f"./{debug_dir}/gt_img.tiff", voxel)
    print(f"uq: {np.unique(seg)}")
