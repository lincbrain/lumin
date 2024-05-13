from typing import Optional, Tuple, List

import dask
import dask.array as da
from dask.diagnostics import ProgressBar

from lsm.distributed.distributed_seg import link_labels


def segment_cellpose(
    image: dask.array,
    diameter: Tuple[float],
    channels: Optional[List[int]] = [[0, 0]],
    model_type: Optional[str] = "nuclei",
    use_anistropy: Optional[bool] = True,
):
    return None


if __name__ == "__main__":
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
    reader = Reader(parse_url(url))
    dask_data = list(reader())[0].data
    scale = 0
    vol_scale = dask_data[scale][0][0]

    seg_vol = segment_cellpose(image=vol_scale)
