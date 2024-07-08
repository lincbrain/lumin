import zarr
import numpy as np
from typing import Optional, List

from lsm.utils.utils import make_compressor, compute_new_shape, ceildiv


def tiff_to_zarr(
    imvol: np.ndarray,
    chunk: Optional[int] = 128,
    max_load: Optional[int] = 128,
    compressor: Optional[str] = "blosc",
    compressor_params: Optional[dict] = None,
    out_fname: Optional[str] = None,
    thickness: Optional[float] = None,
    voxel_size: List[float] = [1, 1, 1],
):
    """
    convert a TIFF file stitched together after distributed
    segmentation, into a OME-ZARR or NIFTI-ZARR pyramid

    args:
    imvol (c, z, y, x): input image volume
    chunk: zarr voxel chunk size
    """

    # default output name
    if out_fname is None:
        raise ValueError(f"please provide an output filename")

    if compressor_params is None:
        compressor_params = {}

    # prepare zarr group
    omz = zarr.storage.DirectoryStore(out_fname)
    omz = zarr.group(store=omz, overwrite=True)

    nchannels = imvol.shape[0]
    dtype = imvol.dtype

    # chunking options
    chunking_options = {
        "chunks": [nchannels] + [chunk] * 3,
        "dimension_separator": r"/",
        "order": "F",
        "dtype": np.dtype(dtype).str,
        "fill_value": None,
        "compressor": make_compressor(compressor, **compressor_params),
    }

    # write 1st level (highest resolution)
    omz.create_dataset(
        "0", data=imvol, shape=[nchannels, *imvol.shape[1:]], **chunking_options
    )
    print(f"Wrote level 0 with shape: {[nchannels, *imvol.shape[1:]]}")

    level = 0
    while any(x > 1 for x in omz[str(level)].shape[-3:]):
        prev_array = omz[str(level)]
        prev_shape = prev_array.shape[-3:]
        level += 1
        print(f"level: {level}")

        new_shape = compute_new_shape(prev_shape)
        print(f"Compute level {level} with shape {new_shape}")

        # initialize new image matrix
        new_array = np.zeros((3, *new_shape))

        nz, ny, nx = prev_array.shape[-3:]
        ncz = ceildiv(nz, max_load)
        ncy = ceildiv(ny, max_load)
        ncx = ceildiv(nx, max_load)

        for cz in range(ncz):
            for cy in range(ncy):
                for cx in range(ncx):
                    print(f"chunk ({cz}, {cy}, {cx}) / ({ncz}, {ncy}, {ncx})", end="\r")

                    dat = prev_array[
                        ...,
                        cz * max_load : (cz + 1) * max_load,
                        cy * max_load : (cy + 1) * max_load,
                        cx * max_load : (cx + 1) * max_load,
                    ]
                    dat = handle_chunk(data=dat, nchannels=nchannels)

                    new_array[
                        ...,
                        cz * max_load // 2 : (cz + 1) * max_load // 2,
                        cy * max_load // 2 : (cy + 1) * max_load // 2,
                        cx * max_load // 2 : (cx + 1) * max_load // 2,
                    ] = dat

        # save
        omz.create_dataset(
            str(level),
            data=new_array,
            shape=[nchannels, *new_shape],
            **chunking_options,
        )

    nblevel = level

    # write ome-zar metadata
    print(f"writing metadata...")
    multiscales = [
        {
            "version": "0.0",
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [],
            "type": "median window 2x2x2",
            "name": "",
        }
    ]
    multiscales[0]["axes"].insert(0, {"name": "c", "type": "channel"})

    voxel_size = list(map(float, reversed(voxel_size)))
    factor = [1] * 3

    for level in range(nblevel):
        shape = omz[str(level)].shape[-3:]
        multiscales[0]["datasets"].append({})
        l = multiscales[0]["datasets"][-1]
        l["path"] = str(level)

        if level > 0:
            shape_prev = omz[str(level - 1)].shape[-3:]
            if shape_prev[0] != shape[0]:
                factor[0] *= 2
            if shape_prev[1] != shape[1]:
                factor[1] *= 2
            if shape_prev[2] != shape[2]:
                factor[2] *= 2

        l["coordinateTransformations"] = [
            {
                "type": "scale",
                "scale": [1.0]
                + [
                    factor[0] * voxel_size[0],
                    factor[1] * voxel_size[1],
                    factor[2] * voxel_size[2],
                ],
            },
            {
                "type": "translation",
                "translation": [0.0]
                + [
                    (factor[0] - 1) * voxel_size[0] * 0.5,
                    (factor[1] - 1) * voxel_size[1] * 0.5,
                    (factor[2] - 1) * voxel_size[2] * 0.5,
                ],
            },
        ]
    multiscales[0]["coordinateTransformations"] = [
        {"scale": [1.0] * 4, "type": "scale"}
    ]

    omz.attrs["multiscales"] = multiscales


def handle_chunk(data, nchannels):
    crop = [0 if x == 1 else x % 2 for x in data.shape[-3:]]
    slicer = [slice(-1) if x else slice(None) for x in crop]
    data = data[(Ellipsis, *slicer)]
    pz, py, px = data.shape[-3:]

    data = data.reshape(
        [
            nchannels,
            max(pz // 2, 1),
            min(pz, 2),
            max(py // 2, 1),
            min(py, 2),
            max(px // 2, 1),
            min(px, 2),
        ]
    )
    data = data.transpose([0, 1, 3, 5, 2, 4, 6])
    data = data.reshape(
        [
            nchannels,
            max(pz // 2, 1),
            max(py // 2, 1),
            max(px // 2, 1),
            -1,
        ]
    )
    data = np.median(data, -1)
    return data


if __name__ == "__main__":
    a = np.random.randn(3, 128, 128, 128)
    tiff_to_zarr(imvol=a, out_fname="test.ome.zarr")
