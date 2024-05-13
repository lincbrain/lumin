import operator
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List

import dask
import dask.array as da
from dask.diagnostics import ProgressBar

import tensorflow as tf
from stardist.models import StarDist3D

from lsm.distributed.distributed_seg import link_labels

# set tensorflow gpu devices
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, enable=True)


def segment_anystar(
    image: dask.array,
    n_tiles: Optional[Tuple[int]] = (1, 1, 1),
    scale: Optional[List[float]] = [1.0, 1.0, 1.0],
    nms_thresh: Optional[float] = 0.3,
    prob_thresh: Optional[float] = 0.5,
    iou_depth: Optional[int] = 2,
    iou_threshold: Optional[float] = 0.7,
):
    """
    segment an entire light sheet volume using anystar
    """
    # create a chunk loader object
    block = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks),
        ),
    )

    # singlular, default boundary condition (is this a good idea?)
    boundary = "reflect"

    segment_chunk = np.empty(image.numblocks[:-1], dtype=object)
    total_blocks = None
    labeled_blocks = np.empty(image.numblocks[:-1], dtype=object)
    total = None
    anisotropy = (25, 36, 25)

    for index, input_block in tqdm(block, desc=f"lazy evaluating chunks..."):
        # normalize input volume to [0, 1]
        input_block = input_block.map_blocks(normalize_volume)

        labeled_block, n = dask.delayed(segment_anystar_chunk, nout=2)(
            chunk=input_block,
            n_tiles=n_tiles,
            nms_thresh=nms_thresh,
            prob_thresh=prob_thresh,
            scale=scale,
        )

        shape = input_block.shape[:-1]
        labeled_block = da.from_delayed(labeled_block, shape=shape, dtype=np.int32)

        n = dask.delayed(np.int32)(n)
        n = da.from_delayed(n, shape=(), dtype=np.int32)

        # count total number of segments
        total = n if total is None else total + n

        block_label_offset = da.where(labeled_block > 0, total, np.int32(0))
        labeled_block += block_label_offset

        # store labeled blocks
        labeled_blocks[index[:-1]] = labeled_block
        total += n

    # put all the blocks together
    block_labeled = da.block(labeled_blocks.tolist())
    anisotropy = da.overlap.coerce_depth(len(anisotropy), anisotropy)

    # check if number of blocks is > 1
    if np.prod(block_labeled.numblocks) > 1:
        iou_depth = da.overlap.coerce_depth(len(anisotropy), iou_depth)

        # check if any axis has greater overlap than anisotropy (pixel spacing)
        if any(iou_depth[ax] > anisotropy[ax] for ax in anisotropy.keys()):
            raise Exception

        # trim blocks to match with overlap depth
        trim_depth = {k: anisotropy[k] - iou_depth[k] for k in anisotropy.keys()}
        block_labeled = da.overlap.trim_internal(
            block_labeled, trim_depth, boundary=boundary
        )

        # link labels across chunks using IOU tracking
        block_labeled = link_labels(
            block_labeled,
            total,
            depth=iou_depth,
            iou_threshold=iou_threshold,
        )

        block_labeled = da.overlap.trim_internal(
            block_labeled, iou_depth, boundary=boundary
        )

    else:
        block_labeled = da.overlap.trim_internal(
            block_labeled, anisotropy, boundary=boundary
        )

    return block_labeled


def segment_anystar_chunk(
    chunk: dask.array,
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
    model.keras_model.trainable = False

    # note that the chunk must be normalized before passing to the model
    labels, _ = model.predict_instances(
        chunk,
        prob_thresh=prob_thresh,
        n_tiles=n_tiles,
        nms_thresh=nms_thresh,
        scale=scale,
    )

    return labels.astype(np.int32), labels.max()


def normalize_volume(vol: dask.array):
    return (vol - vol.min()) / (vol.max() - vol.min())


if __name__ == "__main__":
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
    reader = Reader(parse_url(url))
    dask_data = list(reader())[0].data
    scale = 0
    vol_scale = dask_data[scale][0][0]

    seg_vol = segment_anystar(image=vol_scale)

    # segment on a single GPU, with serial chunks

    from tifffile import imsave

    with ProgressBar():
        with dask.config.set(scheduler="synchronous"):
            seg_vol = seg_vol.compute()
            print(f"seg_vol: {seg_vol.shape}")
            print(f"actual: {vol_scale.shape}")
