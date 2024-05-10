import os
import numpy as np
import nibabel as nib

# import torch
# import torch.nn as nn

import tensorflow as tf
from stardist.models import StarDist3D

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, enable=True)


# class AnyStar(nn.Module):
#    def __init__(self, model_config: dict()):
#        super(AnyStar, self).__init__()
#
#        self.model_config = model_config
#        self.normalize = model_config["normalize"]
#        self.trainable = model_config["trainable"]
#        self.prob_thresh = model_config["prob_thresh"]
#        self.n_tiles = eval(model_config["n_tiles"])  # cast to Tuple[in5]
#        self.nms_thresh = model_config["nms_thresh"]
#        self.scale = model_config["scale"]
#
#        model_type = model_config["model_type"]
#        basedir = model_config["basedir"]
#
#        # instantiate and load model weights
#        self.model = StarDist3D(None, name=model_type, basedir=basedir)
#        self.model.load_weights(name="weights_best.h5")
#
#    def _normalize_vol(self, vol: torch.Tensor):
#        """normalize image volume intensities"""
#        vol = (vol - vol.max()) / (vol.max() - vol.min())
#        return vol
#
#    def forward(self, x: torch.Tensor):
#        self.model.trainable = self.trainable
#        self.model.keras_model.trainable = self.trainable
#
#        x = x.squeeze(0)  # unbatch data
#
#        assert x.ndim == 3, f"AnyStar only supports 3D image volumes"
#
#        x = x.detach().cpu().numpy()
#
#        # normalize intensties if needed
#        if self.normalize:
#            x = self._normalize_vol(vol=x)
#
#        masks, _ = self.model.predict_instances(
#            x,
#            prob_thresh=self.prob_thresh,
#            n_tiles=self.n_tiles,
#            nms_thresh=self.nms_thresh,
#            scale=self.scale,
#        )
#
#        return masks


def get_model(args):
    model_config = {
        "model_type": args.model.model_type,
        "basedir": args.model.basedir,
        "trainable": args.model.trainable,
        "prob_thresh": args.model.prob_thresh,
        "nms_thresh": args.model.nms_thresh,
        "scale": args.model.scale,
        "n_tiles": args.model.n_tiles,
        "normalize": args.model.normalize,
    }

    model = AnyStar(model_config=model_config)

    return model


def normalize_vol(vol):
    print(f"max: {vol.max()}")
    print(f"min: {vol.min()}")
    return (vol - vol.max()) / (vol.max() - vol.min())


def anystar_predict_dask(vol):
    model = StarDist3D(None, name="anystar-mix", basedir="model-weights")
    model.load_weights(name="weights_best.h5")
    model.trainable = False

    labels, _ = model.predict_instances_big(
        img=vol,
        axes="ZYX",
        block_size=[128, 128, 128],
        prob_thresh=0.5,
        min_overlap=[4, 4, 4],
        context=[4, 4, 4],
        n_tiles=(1, 1, 1),
        nms_thresh=0.3,
        scale=[1.0, 1.0, 1.0],
    )

    return labels


if __name__ == "__main__":
    model = StarDist3D(None, name="anystar-mix", basedir="model-weights")
    model.load_weights(name="weights_best.h5")
    model.trainable = False
    model.keras_model.trainable = False

    # fpath = "/om2/user/ckapoor/lsm-segmentation/anystar-data/NucMM-Z/img_0320_0704_0640.nii.gz"
    # img = nib.load(fpath).get_fdata()
    ## print(f"type: {type(img)}")
    ## normalize intensties
    # img_norm = (img - img.max()) / (img.max() - img.min())
    # print(f"norm: {img_norm.shape}")

    # labels, _ = model.predict_instances(
    #    img_norm,
    #    prob_thresh=0.5,
    #    n_tiles=(1, 1, 1),
    #    nms_thresh=0.3,
    #    scale=[1.0, 1.0, 1.0],
    # )

    cmap = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)
    cmap[0] = [0, 0, 0]

    import cv2 as cv

    # for i in [35, 45]:
    #    cv.imwrite(f"{i}_gt.png", img[..., i])
    #    seg = cmap[labels[..., i]]
    #    cv.imwrite(f"{i}_mask.png", seg)

    # visualize some intermediate slices
    # print(f"uq labels: {np.unique(labels)}")
    # labels = labels[..., 40]

    # seg_mask = cmap[labels]
    # print(f"labels shape: {labels.shape}")
    # cv.imwrite(f"anystar_gt.png", img[..., 40])
    # cv.imwrite(f"anystar_mask.png", seg_mask[..., 40])

    from tifffile import imwrite
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    url = "https://dandiarchive.s3.amazonaws.com/zarr/0bda7c93-58b3-4b94-9a83-453e1c370c24/"
    reader = Reader(parse_url(url))
    dask_data = list(reader())[0].data
    scale = 0
    vol_scale = dask_data[scale][0][0]

    # take an arbitrary (64, 64, 64) voxel
    chunk_size = 256
    voxel = vol_scale[
        1500 : 1500 + chunk_size, 2000 : 2000 + chunk_size, 3000 : 3000 + chunk_size
    ].compute()

    print(f"voxel shape: {voxel.shape}")
    # save voxel for sanity check
    imwrite(f"dandiset_subvoxel.tiff", voxel)

    # normalize voxel
    voxel_norm = (voxel - voxel.min()) / ((voxel.max() - voxel.min()))
    labels, _ = model.predict_instances(
        voxel_norm,
        prob_thresh=0.5,
        n_tiles=(1, 1, 1),
        nms_thresh=0.3,
        scale=[0.25, 0.25, 0.25],
    )

    imwrite(f"label_vol_subvoxel.tiff", labels)
    print(f"max: {voxel_norm.max()}")
    print(f"min: {voxel_norm.min()}")

    # from tqdm import tqdm

    ## vol_chunk = vol_norm[1000:1100, ...]
    # labels, _ = model.predict_instances(
    #    vol_norm,
    #    prob_thresh=0.5,
    #    n_tiles=(1, 1, 1),
    #    nms_thresh=0.3,
    #    scale=[1.0, 1.0, 1.0],
    # )
    # print(f"uq: {np.unique(labels)}")

    # from tifffile import imsave

    # for i in [20, 30]:
    #    print(f"saving: {i}")
    #    seg = color_map[labels[..., i]]
    #    imsave(f"gt_img_anystar_{i}.tiff", vol_scale[..., i])
    #    imsave(f"pred_seg_anystar_{i}.tiff", seg[..., i])
    # color_map = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)
    # color_map[0] = [0, 0, 0]

    # for i in [50, 70]:
    #    print(f"saving: {i}")
    #    seg = color_map[labels[i, ...]]
    #    imsave(f"gt_img_anystar_{i+100}.tiff", vol_scale[i + 100, ...])
    #    imsave(f"pred_seg_anystar_{i+100}.tiff", seg[i, ...])

    # labels, _ = model.predict_instances_big(
    #    img=vol_scale,
    #    axes="ZYX",
    #    block_size=[128, 128, 128],
    #    # prob_thresh=0.5,
    #    min_overlap=[4, 4, 4],
    #    context=[4, 4, 4],
    #    # n_tiles=(1, 1, 1),
    #    # nms_thresh=0.3,
    #    scale=[1.0, 1.0, 1.0],
    # )

    # labels, _ = model.predict_instances(
    #    vol_norm,
    #    prob_thresh=0.5,
    #    n_tiles=(1, 1, 1),
    #    nms_thresh=0.3,
    #    scale=[1.0, 1.0, 1.0],
    # )

    # seg_mask = color_map[labels]
    # cv.imwrite(f"anystar_mask.png", labels)
    ## cv.imwrite(f"anystar_mask.png", seg_mask)
    # cv.imwrite(f"anystar_original.png", cv.imread(img_path))
