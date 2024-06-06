import os
import numpy as np
import nibabel as nib

from cellpose import models, io


def cellpose_synthetic(im):
    """
    run cellpose on synthetic data
    """
    model = models.Cellpose(gpu=True, model_type="nuclei")
    im = im[np.newaxis, ...]
    seg, _, _, _ = model.eval(
        im,
        channels=[[0, 0]],
        z_axis=1,
        channel_axis=0,
        do_3D=True,
        anisotropy=4,
        augment=True,
        tile=True,
    )

    return seg


if __name__ == "__main__":
    im_path = "/om2/user/ckapoor/generative_model_steerable_gaussians/outputs/training_images/stack_0003/stack_0003_0000_v13.nii.gz"
    label_path = "/om2/user/ckapoor/generative_model_steerable_gaussians/outputs/training_labels/stack_0003/stack_0003_0000_v13.nii.gz"

    # load images
    im = nib.load(im_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    # predict label
    seg = cellpose_synthetic(im=im)
    print(f"seg uq: {np.unique(seg)}")

    from tifffile import imwrite

    imwrite(f"./cellpose_eval_seg.tiff", seg)
    imwrite(f"./cellpose_gt_seg.tiff", label)
    imwrite(f"./cellpose_gt_vol.tiff", im)
