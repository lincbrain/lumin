import os
import operator
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List

import dask
import dask.array as da
from dask.diagnostics import ProgressBar

from cellpose import models

from lsm.processing.normalize import normalize_image
from lsm.distributed.distributed_seg import link_labels


def segment_cellpose(
    image: dask.array,
    debug: Optional[bool] = False,
    channels: Optional[List[int]] = [[0, 0]],
    model_type: Optional[str] = "nuclei",
    boundary: Optional[str] = "reflect",
    diameter: Tuple[float] = None,
    chunk: Optional[int] = None,
    use_anisotropy: Optional[bool] = True,
    iou_depth: Optional[int] = 2,
    iou_threshold: Optional[float] = 0.7,
):

    diameter_yx = diameter[1]
    anisotropy = diameter[0] / diameter[1] if use_anisotropy else None

    image = da.asarray(image)

    # for re-chunking/stitching analyses
    if chunk is None:
        image = image.rechunk({-1: -1})
    else:
        image = image.rechunk({0: chunk, 1: chunk, 2: chunk, 3: -1})

    # define depth for stitching voxel blocks
    depth = tuple(np.ceil(diameter).astype(np.int64))

    # boundary condition options for chunked dask arrays
    boundary = boundary

    # no chunking along channel direction
    image = da.overlap.overlap(image, depth + (0,), boundary)

    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks),
        ),
    )

    labeled_blocks = np.empty(image.numblocks[:-1], dtype=object)
    # initialize empty "grid" for chunks
    if debug:
        unlabeled_blocks = np.empty(image.numblocks[:-1], dtype=object)

    total = None
    for index, input_block in tqdm(block_iter, desc="lazy computing chunks..."):

        labeled_block, n = dask.delayed(segment_cellpose_chunk, nout=2)(
            chunk=input_block,
            index=index,
            channels=channels,
            model_type=model_type,
            diameter_yx=diameter_yx,
            anisotropy=anisotropy,
        )

        shape = input_block.shape[:-1]
        labeled_block = da.from_delayed(labeled_block, shape=shape, dtype=np.int32)

        n = dask.delayed(np.int32)(n)
        n = da.from_delayed(n, shape=(), dtype=np.int32)

        total = n if total is None else total + n

        block_label_offset = da.where(labeled_block > 0, total, np.int32(0))
        labeled_block += block_label_offset

        labeled_blocks[index[:-1]] = labeled_block
        total += n
        print(f"labeled shape: {labeled_block.shape}")

        if debug:
            # do the same thing, but assign the same label to *every* chunk
            # here, we change the image to be 4D, to account for a newly
            # introduced color channel
            unlabeled_block = labeled_block
            colored_chunk = unlabeled_block.copy()
            nz_mask = (unlabeled_block != 0).astype(
                np.int32
            )  # find non-zero pixel locations
            colored_chunk = np.zeros(
                unlabeled_block.shape + (3,), dtype=np.int32
            )  # add a color channel
            color = np.random.randint(0, 256, size=(3,))
            colored_chunk[nz_mask] = color
            # unlabeled_block[nz_indices + (slice(None),)] = random_colors
            # print(f"colored chunk: {colored_chunk.shape}")
            unlabeled_blocks[index[:-1]] = colored_chunk

    # put all blocks together
    block_labeled = da.block(labeled_blocks.tolist())

    if debug:
        block_unlabeled = da.block(unlabeled_blocks.tolist())

    depth = da.overlap.coerce_depth(len(depth), depth)

    if np.prod(block_labeled.numblocks) > 1:
        iou_depth = da.overlap.coerce_depth(len(depth), iou_depth)

        if any(iou_depth[ax] > depth[ax] for ax in depth.keys()):
            raise Exception

        trim_depth = {k: depth[k] - iou_depth[k] for k in depth.keys()}
        block_labeled = da.overlap.trim_internal(
            block_labeled, trim_depth, boundary=boundary
        )

        # trim excess, due to reflections
        if debug:
            block_unlabeled = da.overlap.trim_internal(
                block_unlabeled, trim_depth, boundary=boundary
            )

        block_labeled = link_labels(
            block_labeled, total, iou_depth, iou_threshold=iou_threshold
        )

        block_labeled = da.overlap.trim_internal(
            block_labeled, iou_depth, boundary=boundary
        )

    else:
        block_labeled = da.overlap.trim_internal(
            block_labeled, depth, boundary=boundary
        )
        if debug:
            block_unlabeled = da.overlap.trim_internal(
                block_unlabeled, depth, boundary=boundary
            )

    if debug:
        return block_labeled, block_unlabeled

    return block_labeled


def segment_cellpose_chunk(
    chunk: dask.array,
    index: Optional[int],
    channels: Optional[List] = [[0, 0]],
    model_type: Optional[str] = "nuclei",
    diameter_yx: Optional[float] = 7.5,
    anisotropy: Optional[float] = 4,
):
    np.random.seed(index)

    model = models.Cellpose(gpu=True, model_type=model_type)

    seg, _, _, _ = model.eval(
        chunk,
        channels=channels,
        z_axis=0,
        channel_axis=3,
        diameter=diameter_yx,
        do_3D=True,
        anisotropy=anisotropy,
        augment=True,
        tile=True,
    )

    return seg.astype(np.int32), seg.max()


if __name__ == "__main__":
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
    reader = Reader(parse_url(url))
    dask_data = list(reader())[0].data
    scale = 0
    vol_scale = dask_data[scale][0]
    vol_scale = np.transpose(vol_scale, (1, 2, 3, 0))

    chunk_size = 64
    voxel = vol_scale[
        1000 : 1000 + chunk_size, 650 : 650 + chunk_size, 3500 : 3500 + chunk_size
    ]

    from tifffile import imsave

    debug_dir = f"./chunk64_debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # chunks = [20, 25, 30, 35, 40, 45, 50, 55]
    chunks = [55]

    for chunk in tqdm(chunks):
        seg_vol, debug_vol = segment_cellpose(
            image=voxel, diameter=(7.5, 7.5, 7.5), debug=True, chunk=chunk
        )

        with ProgressBar():
            with dask.config.set(scheduler="synchronous"):
                seg_vol = seg_vol.compute()
                imsave(f"./{debug_dir}/seg_cellpose_chunk{chunk}.tiff", seg_vol)
                imsave(f"./{debug_dir}/debug_cellpose_chunk{chunk}.tiff", debug_vol)
