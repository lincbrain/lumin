import os
import time
import operator
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List

import dask
import dask.array as da
from dask.diagnostics import ProgressBar

from stardist.models import StarDist3D

from lsm.distributed.distributed_seg import link_labels


def segment(
    image: dask.array,
    model_folder: str,
    model_name: str,
    weight_name: str,
    prob_thresh: float = 0.67,
    nms_thresh: float = 0.3,
    scale: List[float] = [1.0, 1.0, 1.0],
    debug: Optional[bool] = False,
    boundary: Optional[str] = "reflect",
    diameter: List[float] = None,
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

    for index, input_block in tqdm(
        block_iter, desc="lazy computing chunks using anystar..."
    ):

        labeled_block, n = dask.delayed(segment_anystar_chunk, nout=2)(
            chunk=input_block,
            index=index,
            scale=scale,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            model_name=model_name,
            model_folder=model_folder,
            weight_name=weight_name,
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
        # print(f"labeled shape: {labeled_block.shape}")

        if debug:
            # do the same thing, but assign the same label to *every* chunk
            # here, we change the image to be 4D, to account for a newly
            # introduced color channel
            unlabeled_block = labeled_block
            colored_chunk = unlabeled_block.copy()
            nz_mask = (unlabeled_block != 0).astype(
                np.int32
            )  # find non-zero pixel locations
            # colored_chunk = np.zeros(
            #    unlabeled_block.shape + (3,), dtype=np.int32
            # )  # add a color channel
            # color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
            color = np.random.randint(0, 256, dtype=np.uint8)
            colored_chunk[nz_mask] = color
            unlabeled_blocks[index[:-1]] = colored_chunk

    # put all blocks together
    tick = time.time()

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

    tock = time.time()
    print(f"stitching took: {tock - tick}")

    if debug:
        return block_labeled, block_unlabeled

    return block_labeled


def segment_anystar_chunk(
    chunk: dask.array,
    index: Optional[int],
    model_folder: str,
    model_name: str,
    weight_name: str,
    prob_thresh: float,
    nms_thresh: float,
    scale: List[float],
    diameter_yx: Optional[float] = 7.5,
    anisotropy: Optional[float] = 4,
):
    np.random.seed(index)

    model = StarDist3D(None, name=model_name, basedir=model_folder)
    model.load_weights(name=weight_name)
    model.trainable = False
    model.keras_model.trainable = False

    # normalize chunk to [0, 1] before running inference
    tick = time.time()
    upper = np.percentile(chunk, 99.9)
    chunk = np.clip(chunk, 0, upper)
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
    chunk = chunk[..., 0]

    print(f"tiling with stardist-based model...")
    seg, _ = model.predict_instances(
        chunk,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=None,
        # n_tiles=chunk.shape,
        scale=scale,
    )
    tock = time.time()
    print(f"seg of a chunk took: {tock - tick}")

    return seg.astype(np.int32), seg.max()


if __name__ == "__main__":
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
    tick = time.time()
    reader = Reader(parse_url(url))
    dask_data = list(reader())[0].data
    tock = time.time()
    print(f"streaming: {tock - tick}")
    scale = 0
    vol_scale = dask_data[scale][0]

    vol_scale = np.transpose(vol_scale, (1, 2, 3, 0))  # (c, z, y, x) -> (z, y, x, c)

    chunk_size = 128
    voxel = vol_scale[
        1000 : 1000 + chunk_size, 650 : 650 + chunk_size, 3500 : 3500 + chunk_size
    ]

    from tifffile import imwrite

    # model_name = "gaussian_steerable_run_250k"
    model_name = "spherical_steerable_run"
    model_folder = "../../models"
    weight_name = "weights_best.h5"

    from dask.distributed import Client, performance_report

    chunk = 64
    seg_vol, debug_vol = segment(
        image=voxel,
        diameter=(7.5 * 4, 7.5, 7.5),
        model_name=model_name,
        model_folder=model_folder,
        weight_name=weight_name,
        debug=True,
    )
    seg_vol.compute()
    debug_vol.compute()
    imwrite(f"debug_anystar.tiff", debug_vol)

    # chunks = [8, 16, 32, 64]
    # model_name = "anystar-mix"
    # model_folder = "models"
    # model = StarDist3D(None, name=model_name, basedir=model_folder)
    # model.load_weights(name="weights_best.h5")
    # model.trainable = False
    # model.keras_model.trainable = False

    ## normalize voxel

    # x = voxel.compute()
    # upper = np.percentile(x, 99.9)
    # x = np.clip(x, 0, upper)
    # x = (x - x.min()) / (x.max() - x.min())
    # x = x[..., 0]

    # prob, _ = model.predict(img=x)
    # imwrite(f"./probmap_spherical.tiff", prob)
    # seg, _ = model.predict_instances(
    #    x,
    #    prob_thresh=0.67,
    #    nms_thresh=0.3,
    #    n_tiles=None,
    #    scale=[1.0, 1.0, 1.0],
    # )

    # for chunk in tqdm(chunks):
    #    seg_vol = segment(
    #        image=voxel,
    #        diameter=(7.5 * 4, 7.5, 7.5),
    #        chunk=chunk,
    #        model_folder=model_folder,
    #        model_name=model_name,
    #        weight_name=weight_name,
    #    )

    #    with ProgressBar():
    #        with dask.config.set(scheduler="synchronous"):
    #            seg_vol = seg_vol.compute()
    #            imwrite(f"./anystar_chunk{chunk}.tiff", seg_vol)


# if __name__ == "__main__":
#    from ome_zarr.io import parse_url
#    from ome_zarr.reader import Reader
#
#    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
#    reader = Reader(parse_url(url))
#    dask_data = list(reader())[0].data
#    scale = 0
#    vol_scale = dask_data[scale][0]
#    print(f"vs shape: {vol_scale.shape}")
#
#    vol_scale = np.transpose(vol_scale, (1, 2, 3, 0))  # (c, z, y, x) -> (z, y, x, c)
#
#    chunk_size = 64
#    voxel = vol_scale[
#        1000 : 1000 + chunk_size, 650 : 650 + chunk_size, 3500 : 3500 + chunk_size
#    ]
#
#    from tifffile import imsave
#
#    debug_dir = f"./chunk64_stardist_debug"
#    if not os.path.exists(debug_dir):
#        os.makedirs(debug_dir)
#
#    # model_name = "gaussian_steerable_run"
#    model_name = "anystar-mix"
#    model_folder = "models"
#    model = StarDist3D(None, name=model_name, basedir=model_folder)
#    model.load_weights(name="weights_best.h5")
#    model.trainable = False
#    model.keras_model.trainable = False
#
#    # normalize voxel
#
#    x = voxel.compute()
#    upper = np.percentile(x, 99.9)
#    x = np.clip(x, 0, upper)
#    x = (x - x.min()) / (x.max() - x.min())
#    x = x[..., 0]
#
#    seg, _ = model.predict_instances(
#        x,
#        prob_thresh=0.67,
#        nms_thresh=0.3,
#        n_tiles=None,
#        scale=[1.0, 1.0, 1.0],
#    )
#
#    seg = np.asarray(seg)
#    from tifffile import imwrite
#
#    imwrite(f"./{debug_dir}/pred_anystar_seg.tiff", seg)
#    imwrite(f"./{debug_dir}/gt_img.tiff", voxel)
#    print(f"uq: {np.unique(seg)}")
