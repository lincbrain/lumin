import operator
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List

import dask
import dask.array as da

import tensorflow as tf
from stardist.models import StarDist3D

# set tensorflow gpu devices
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, enable=True)


def segment_anystar(
    image,
    n_tiles: Optional[Tuple[int]] = (1, 1, 1),
    scale: Optional[List[float]] = [1.0, 1.0, 1.0],
    nms_thresh: Optional[float] = 0.3,
    prob_thresh: Optional[float] = 0.5,
    iou_depth: Optional[int] = 2,
    iou_threshold: Optional[float] = 0.7,
):
    """
    segment an entire light sheet volume
    """
    # lazy normalize volume

    # create a chunk loader object
    block = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks),
        ),
    )

    labeled_blocks = np.empty(image.numblocks[:-1], dtype=object)
    total_blocks = None

    for idx, chunk in tqdm(block, desc=f"processing chunks..."):
        # normalize volume

        seg, n_seg = dask.delayed(segment_anystar_chunk, nout=2)(
            chunk=chunk,
            n_tiles=n_tiles,
            nms_thresh=nms_thresh,
            prob_thresh=prob_thresh,
            scale=scale,
        )
        shape = chunk.shape[:-1]
        seg = da.from_delayed(seg, shape=shape, dtype=np.int32)

        n_seg = dask.delayed(np.int32)(n_seg)
        n_seg = da.from_delayed(n_seg, shape=(), dtype=np.int32)

        total_blocks = n_seg if total_blocks is None else total_blocks + n_seg

        # account for chunk offsets
        chunk_label_offset = da.where(seg > 0, total_blocks, np.int32(0))


def segment_anystar_chunk(
    chunk,
    n_tiles: Tuple[int],
    nms_thresh: float,
    prob_thresh: float,
    scale: List[float],
):
    """
    segment a dask array chunk from the entire volume
    """
    model = StarDist3D(None, name="anystar-mix", basedir="model-weights")
    model.load_weights(name="weights_best.h5")
    model.trainable = False
    model.keras_mdoel.trainable = False

    print(f"evaluating chunk...")
    # note that the chunk must be normalized before passing to the model
    labels, _ = model.predict_instances(
        chunk,
        prob_thresh=prob_thresh,
        n_tiles=n_tiles,
        nms_thresh=nms_thresh,
        scale=scale,
    )
    print(f"done evaluating chunk!")
    print(f"seg: {np.unique(labels)}")

    return labels.astype(np.int32), labels.max()


def normalize_volume(vol):
    return (vol - vol.min()) / (vol.max() - vol.min())


if __name__ == "__main__":
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
    reader = Reader(parse_url(url))
    dask_data = list(reader())[0].data
    scale = 0
    vol_scale = dask_data[scale][0][0]

    segment_anystar(image=vol_scale)
