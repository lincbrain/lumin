import os
import functools
import logging
import operator
from tqdm import tqdm

from tifffile import imwrite

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np

GLOBAL_COUNTER = 0


class DistSegError(Exception):
    """Error in image segmentation."""


try:
    from dask_image.ndmeasure._utils import _label
    from sklearn import metrics as sk_metrics
except ModuleNotFoundError as e:
    raise DistSegError(
        "Install 'cellpose[distributed]' for distributed segmentation dependencies"
    ) from e


logger = logging.getLogger(__name__)


def segment(
    image,
    channels,
    model_type,
    diameter,
    chunk=None,
    fast_mode=False,
    use_anisotropy=True,
    iou_depth=2,
    iou_threshold=0.7,
):
    assert image.ndim == 4, image.ndim
    assert image.shape[-1] in {1, 2}, image.shape
    assert diameter[1] == diameter[2], diameter

    diameter_yx = diameter[1]
    anisotropy = diameter[0] / diameter[1] if use_anisotropy else None

    image = da.asarray(image)
    # image = image.rechunk({-1: -1})  # color channel is chunked together
    # change chunking

    if chunk is None:
        # only chunk channel dimension
        image = image.rechunk({-1: -1})
    else:
        image = image.rechunk({0: chunk, 1: chunk, 2: chunk, 3: -1})

    depth = tuple(np.ceil(diameter).astype(np.int64))
    boundary = "reflect"

    # No chunking in channel direction
    image = da.overlap.overlap(image, depth + (0,), boundary)

    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks),
        ),
    )

    labeled_blocks = np.empty(image.numblocks[:-1], dtype=object)
    total = None
    for index, input_block in tqdm(block_iter, desc="lazy computing chunks..."):
        labeled_block, n = dask.delayed(segment_chunk, nout=2)(
            input_block,
            channels,
            model_type,
            diameter_yx,
            anisotropy,
            fast_mode,
            index,
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

    # Put all the blocks together
    block_labeled = da.block(labeled_blocks.tolist())

    depth = da.overlap.coerce_depth(len(depth), depth)

    if np.prod(block_labeled.numblocks) > 1:
        iou_depth = da.overlap.coerce_depth(len(depth), iou_depth)

        if any(iou_depth[ax] > depth[ax] for ax in depth.keys()):
            raise DistSegError("iou_depth (%s) > depth (%s)" % (iou_depth, depth))

        trim_depth = {k: depth[k] - iou_depth[k] for k in depth.keys()}
        block_labeled = da.overlap.trim_internal(
            block_labeled, trim_depth, boundary=boundary
        )
        block_labeled = link_labels(
            block_labeled,
            total,
            iou_depth,
            iou_threshold=iou_threshold,
        )

        block_labeled = da.overlap.trim_internal(
            block_labeled, iou_depth, boundary=boundary
        )

    else:
        block_labeled = da.overlap.trim_internal(
            block_labeled, depth, boundary=boundary
        )

    return block_labeled


def segment_chunk(
    chunk,
    channels,
    model_type,
    diameter_yx,
    anisotropy,
    fast_mode,
    index,
):
    """Perform segmentation on an individual chunk."""
    np.random.seed(index)

    from cellpose import models

    model = models.Cellpose(gpu=True, model_type=model_type)

    logger.info("Evaluating model")
    segments, _, _, _ = model.eval(
        chunk,
        channels=channels,
        z_axis=0,
        channel_axis=3,
        diameter=diameter_yx,
        do_3D=True,
        anisotropy=anisotropy,
        augment=not fast_mode,
        tile=not fast_mode,
    )
    logger.info("Done segmenting chunk")

    return segments.astype(np.int32), segments.max()


def link_labels(block_labeled, total, depth, iou_threshold=1):
    """
    Build a label connectivity graph that groups labels across blocks,
    use this graph to find connected components, and then relabel each
    block according to those.
    """
    label_groups = label_adjacency_graph(block_labeled, total, depth, iou_threshold)
    new_labeling = _label.connected_components_delayed(label_groups)
    return _label.relabel_blocks(block_labeled, new_labeling)


def label_adjacency_graph(labels, nlabels, depth, iou_threshold):
    all_mappings = [da.empty((2, 0), dtype=np.int32, chunks=1)]

    slices_and_axes = get_slices_and_axes(labels.chunks, labels.shape, depth)
    for face_slice, axis in slices_and_axes:
        face = labels[face_slice]
        mapped = _across_block_iou_delayed(face, axis, iou_threshold)
        all_mappings.append(mapped)

    i, j = da.concatenate(all_mappings, axis=1)
    result = _label._to_csr_matrix(i, j, nlabels + 1)
    return result


def _across_block_iou_delayed(face, axis, iou_threshold):
    """Delayed version of :func:`_across_block_label_grouping`."""
    _across_block_label_grouping_ = dask.delayed(_across_block_label_iou)
    grouped = _across_block_label_grouping_(face, axis, iou_threshold)
    return da.from_delayed(grouped, shape=(2, np.nan), dtype=np.int32)


def _across_block_label_iou(face, axis, iou_threshold):
    unique = np.unique(face)
    face0, face1 = np.split(face, 2, axis)

    intersection = sk_metrics.confusion_matrix(face0.reshape(-1), face1.reshape(-1))
    sum0 = intersection.sum(axis=0, keepdims=True)
    sum1 = intersection.sum(axis=1, keepdims=True)

    # Note that sum0 and sum1 broadcast to square matrix size.
    union = sum0 + sum1 - intersection

    # Ignore errors with divide by zero, which the np.where sets to zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(intersection > 0, intersection / union, 0)

    labels0, labels1 = np.nonzero(iou >= iou_threshold)

    labels0_orig = unique[labels0]
    labels1_orig = unique[labels1]
    grouped = np.stack([labels0_orig, labels1_orig])

    valid = np.all(grouped != 0, axis=0)  # Discard any mappings with bg pixels
    return grouped[:, valid]


def get_slices_and_axes(chunks, shape, depth):
    ndim = len(shape)
    depth = da.overlap.coerce_depth(ndim, depth)
    slices = da.core.slices_from_chunks(chunks)
    slices_and_axes = []
    for ax in range(ndim):
        for sl in slices:
            if sl[ax].stop == shape[ax]:
                continue
            slice_to_append = list(sl)
            slice_to_append[ax] = slice(
                sl[ax].stop - 2 * depth[ax], sl[ax].stop + 2 * depth[ax]
            )
            slices_and_axes.append((tuple(slice_to_append), ax))
    return slices_and_axes


if __name__ == "__main__":
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
    reader = Reader(parse_url(url))
    dask_data = list(reader())[0].data
    scale = 0
    vol_scale = dask_data[scale][0]
    vol_scale = np.transpose(vol_scale, (1, 2, 3, 0))
    # print(f"vs: {vol_scale.shape}")

    chunk_size = 128
    voxel = vol_scale[
        1000 : 1000 + chunk_size, 650 : 650 + chunk_size, 3500 : 3500 + chunk_size
    ]
    seg_vol = segment(
        image=voxel,
        channels=[[0, 0]],
        model_type="nuclei",
        diameter=(7.5, 7.5, 7.5),
    )

    from tifffile import imsave

    # chunks = [100, 110, 120, 130, 140, 150]
    chunks = [20, 25, 30, 35, 40, 45, 50, 55]

    expdir = "./chunk_experiments"
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    # save gt volume
    imsave(f"{expdir}/chunk_128_gt.tiff", voxel.compute())

    for chunk in tqdm(chunks):
        seg_vol = segment(
            image=voxel,
            channels=[[0, 0]],
            model_type="nuclei",
            chunk=chunk,
            diameter=(7.5 * 4, 7.5, 7.5),
        )

        with ProgressBar():
            with dask.config.set(scheduler="synchronous"):
                seg_vol = seg_vol.compute()
                print(f"saving experiment with chunk = {chunk}")
                imsave(f"./{expdir}/chunk{chunk}_cellpose.tiff", seg_vol)

    # with ProgressBar():
    #    with dask.config.set(scheduler="synchronous"):
    #        seg_vol = seg_vol.compute()
    #        print(f"seg shape: {seg_vol.shape}")
    #        imsave(f"distributed_cellpose_chunk100.tiff", seg_vol)
    #        imsave(f"distributed_gt.tiff", voxel.compute())
