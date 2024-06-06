import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional

from scipy.ndimage import gaussian_filter

import nibabel as nib

from itertools import product
from perlin_numpy import generate_perlin_noise_3d


def calculate_distance(center_coords, grid_size: Optional[int] = 128):
    # assign n_vox x n_spheres array:
    distances = np.zeros((grid_size**3, len(center_coords)))

    # list all voxel indices
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    z = np.arange(grid_size)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    vox_coord = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # computes distances using broadcasting
    # p.s.: this is now much faster than looping :)
    distances = np.sqrt(
        np.sum((vox_coord[:, np.newaxis, :] - center_coords) ** 2, axis=2)
    )

    return distances


def initial_label_generator(
    grid_size: Optional[int] = 128,
    r_mean: Optional[int] = 12,
    sigma: Optional[float] = 1.0,
):
    Nz, Ny, Nx = (grid_size,) * 3

    xx = np.arange(0, grid_size, 2 * r_mean)[1:-1]
    z, y, x = np.meshgrid(xx, xx, xx, indexing="ij")  # sphere center coordinates

    # random translation of sphere centers
    points = np.stack([z.flatten(), y.flatten(), x.flatten()]).T
    points = (points).astype(np.float32)
    points_perturbed = points + 0.5 * r_mean * np.random.uniform(-1, 1, points.shape)

    # randomly drop spheres
    ind = np.arange(len(points_perturbed))
    np.random.shuffle(ind)

    ind_keep = ind[: np.random.randint(2 * len(ind) // 3, len(ind))]  # indices to keep
    points_perturbed_kept = points_perturbed[ind_keep]

    # randomly scale sphere radii
    rads = r_mean * np.random.uniform(0.6, 1.2, len(points))
    rads_kept = rads[ind_keep]  # randomly drop the same radii

    # compute spheres distance matrix
    dist_mtx = calculate_distance(points_perturbed_kept)

    # sample perlin noise
    noise_sample = generate_perlin_noise_3d((Nz, Ny, Nx), res=(8, 8, 8))

    # corrupt distance matrix using perlin noise
    corr_dist_mtx = dist_mtx + 0.9 * r_mean * noise_sample.flatten()[:, np.newaxis]

    # label assignment to all pixels
    labelmap = np.zeros(grid_size**3, dtype=np.uint16)
    for j in tqdm(range(dist_mtx.shape[0]), desc=f"assigning labels..."):
        finder = np.where(corr_dist_mtx[j, :] < rads_kept)[0]
        if len(finder) > 0:
            # assign to closes label in case of ambiguity
            value = finder[np.argmin(corr_dist_mtx[j, finder])]
            labelmap[j] = value + 1

    labelmap = np.reshape(labelmap, (grid_size, grid_size, grid_size))

    return labelmap


if __name__ == "__main__":
    np.random.seed(6969)

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--grid_size",
        type=int,
        default=128,
        help="side-length in voxels of the synthesized volume",
    )
    parser.add_argument(
        "--r_mean",  # one or both r_max & r_min have to be None to use this
        type=int,
        default=12,
        help="average radius in voxels of initial spheres",
    )
    parser.add_argument(
        "--r_max",
        type=int,
        default=None,
        help="Used if radius randomized. Specify min sphere radius in voxels",
    )
    parser.add_argument(
        "--r_min",
        type=int,
        default=None,
        help="Used if radius randomized. Specify min sphere radius in voxels",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=27,
        help="number of label maps to synthesize",
    )

    args = parser.parse_args()
    root_dir = f"/om2/user/ckapoor/generative_model_steerable_harmonics/"

    for i in range(args.n_images):
        print("Generating label {}/{}".format(i + 1, args.n_images))
        if args.r_min is None or args.r_max is None:
            label = initial_label_generator(args.grid_size, args.r_mean)
        else:
            radius = np.random.randint(args.r_min, args.r_max + 1)
            label = initial_label_generator(args.grid_size, radius)
        os.makedirs(f"{root_dir}/outputs/initial_labels/", exist_ok=True)
        nib.save(
            nib.Nifti1Image(label, np.eye(4)),
            f"{root_dir}/outputs/initial_labels/stack_{i:04d}.nii.gz",
        )
