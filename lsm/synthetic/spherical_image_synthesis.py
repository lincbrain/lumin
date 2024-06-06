import numpy as np
import cv2 as cv
import argparse
import nibabel as nib
from tqdm import tqdm
import tensorflow as tf
import glob, os, natsort


from scipy.ndimage import rotate
from scipy.ndimage import convolve
from scipy.special import sph_harm

from skimage.transform import resize
from skimage.measure import label as unique_label

from lsm.synthetic.synthseg_utils import (
    draw_value_from_distribution,
    SampleConditionalGMM,
)

import voxelmorph as vxm
import neurite as ne


def compute_spherical_harmonics(l, m, phi, theta):
    # compute harmonics of degree l, order m at (theta, phi)
    return sph_harm(m, l, phi, theta)


def spherical_to_cartesian(r, theta, phi):
    # convert spherical to cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def gaussian_3d_kernel(size, sigma):
    # sample a random 3d gaussian kernel
    x = np.random.normal(0, 1, size[0])
    y = np.random.normal(0, 1, size[1])
    z = np.random.normal(0, 1, size[2])

    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    distance = np.sqrt(x**2 + y**2 + z**2)
    kernel = np.exp(-(distance**2 / (2.0 * sigma**2)))
    kernel = kernel / np.sum(kernel)
    return kernel, (x, y, z)


def rotate_3d_kernel(kernel, angles):
    rotated_kernel = rotate(kernel, angle=angles[0], axes=(1, 2), reshape=False)
    rotated_kernel = rotate(rotated_kernel, angle=angles[1], axes=(0, 2), reshape=False)
    rotated_kernel = rotate(rotated_kernel, angle=angles[2], axes=(0, 1), reshape=False)
    return rotated_kernel


def project_spherical_harmonic(size, sigma, l, m, angles):
    # compute a (random) 3d gaussian kernel
    gaussian, (x, y, z) = gaussian_3d_kernel(size=size, sigma=sigma)

    # steer the gaussian
    gaussian = rotate_3d_kernel(kernel=gaussian, angles=angles)

    # cartesian to spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    # ensure range of arccos is [-1, 1]
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.sign(y) * np.arccos(np.clip(x / np.sqrt(x**2 + y**2), -1, 1))

    harmonic_basis = compute_spherical_harmonics(l=l, m=m, phi=phi, theta=theta).real

    projected_gaussian = (
        harmonic_basis * gaussian
    )  # project gaussians to a spherical harmonic basis

    return projected_gaussian


if __name__ == "__main__":
    np.random.seed(6969)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=10,
        help="number of GMM images to sample from each synthesized label map",
    )
    args = parser.parse_args()
    nimgs = args.n_imgs

    root_dir = f"/om2/user/ckapoor/generative_model_steerable_harmonics/"

    segpath_base = f"{root_dir}/outputs/initial_labels/"  # step 1 initial synth labels
    imgpath_base = f"{root_dir}/outputs/gmm_perlin_images/"  # location for step2 imgs
    labpath_base = f"{root_dir}/outputs/gmm_perlin_labels/"  # location for step2 labs
    segs = natsort.natsorted(glob.glob(segpath_base + "/*.nii.gz"))

    for i in tqdm(range(len(segs))):
        print("image {}/{}".format(i + 1, len(segs)))

        # For each initial label map, create folders to store synth images:
        os.makedirs(
            imgpath_base + "/{}".format(os.path.basename(segs[i])[:-7]),
            exist_ok=True,
        )
        os.makedirs(
            labpath_base + "/{}".format(os.path.basename(segs[i])[:-7]),
            exist_ok=True,
        )
        complete_labels = nib.load(segs[i]).get_fdata()

        # for each label map, generate initial gmmXperlin images:
        for j in tqdm(range(nimgs)):
            im_path = imgpath_base + "/{}/{}_{:04}.nii.gz".format(
                os.path.basename(segs[i])[:-7], os.path.basename(segs[i])[:-7], j
            )
            lab_path = labpath_base + "/{}/{}_{:04}.nii.gz".format(
                os.path.basename(segs[i])[:-7], os.path.basename(segs[i])[:-7], j
            )

            if os.path.exists(im_path) and os.path.exists(lab_path):
                print(f"already synthesized, skipping iteration...")
                continue

            # First, we pad and resize back in order to simulate varying object
            # densities.

            # Mode with which to pad initial label maps:
            # Constant just zero pads, reflect adds more instances
            randmode = np.random.choice(["constant", "reflect"])

            # Amount to pad initial label maps along each axis:
            randpad_x = np.random.choice([0, 32, 64, 96])
            randpad_y = np.random.choice([0, 32, 64, 96])
            randpad_z = np.random.choice([0, 32, 64, 96])

            # Pad:
            current_labels = np.pad(
                complete_labels,
                [
                    [randpad_x, randpad_x],
                    [randpad_y, randpad_y],
                    [randpad_z, randpad_z],
                ],
                mode=randmode,
            )

            # Resize:
            current_labels = resize(
                current_labels,
                (128, 128, 128),
                preserve_range=True,
                anti_aliasing=False,
                order=0,
            )
            # Make sure each label is unique:
            current_labels = unique_label(current_labels).astype(np.uint16)

            # Second, we begin the GMM procedure.
            # Sample means and standard deviations for each object:
            means = draw_value_from_distribution(
                None,
                len(np.unique(current_labels)),
                "uniform",
                125,
                100,
                positive_only=True,
            )
            stds = draw_value_from_distribution(
                None,
                len(np.unique(current_labels)),
                "uniform",
                15,
                10,
                positive_only=True,
            )

            # Background processing. This generates 'AS-Mix'.
            backgnd_mode = np.random.choice(["plain", "rand", "perlin"])
            if backgnd_mode == "plain":
                min_mean = means.min() * np.random.rand(1)
                means[0] = 1.0 * min_mean
                stds[0] = np.random.uniform(0.0, 5.0, 1)
            elif backgnd_mode == "perlin":
                # Inspired by the sm-shapes generative model from
                # https://martinos.org/malte/synthmorph/
                n_texture_labels = np.random.randint(1, 21)
                idx_texture_labels = np.arange(0, n_texture_labels, 1)
                im = ne.utils.augment.draw_perlin(
                    out_shape=(128, 128, 128, n_texture_labels),
                    scales=(32, 64),
                    max_std=1,
                )
                try:
                    warp = ne.utils.augment.draw_perlin(
                        out_shape=(128, 128, 128, n_texture_labels, 3),
                        scales=(16, 32, 64),
                        max_std=16,
                    )
                except:
                    continue

                # Transform and create background label map.
                im = vxm.utils.transform(im, warp)
                background_struct = np.uint8(tf.argmax(im, axis=-1))

                # Background moments for GMM:
                background_means = draw_value_from_distribution(
                    None,
                    len(np.unique(idx_texture_labels)),
                    "uniform",
                    125,
                    100,
                    positive_only=True,
                )
                background_stds = draw_value_from_distribution(
                    None,
                    len(np.unique(idx_texture_labels)),
                    "uniform",
                    15,
                    10,
                    positive_only=True,
                )

            # Sample perlin noise for cell texture here
            randperl = ne.utils.augment.draw_perlin(
                out_shape=(128, 128, 128, 1),
                scales=(2, 4, 8, 16, 32),
                max_std=5.0,
            )[..., 0].numpy()
            randperl = (randperl - randperl.min()) / (randperl.max() - randperl.min())

            # Create foreground:
            synthlayer = SampleConditionalGMM(np.unique(current_labels))
            synthimage = synthlayer(
                [
                    tf.convert_to_tensor(
                        current_labels[np.newaxis, ..., np.newaxis],
                        dtype=tf.float32,
                    ),
                    tf.convert_to_tensor(
                        means[tf.newaxis, ..., tf.newaxis],
                        dtype=tf.float32,
                    ),
                    tf.convert_to_tensor(
                        stds[tf.newaxis, ..., tf.newaxis],
                        dtype=tf.float32,
                    ),
                ]
            )[0, ..., 0].numpy()

            # Use multiplicative Perlin noise on foreground:
            synthimage[current_labels > 0] = (
                synthimage[current_labels > 0] * randperl[current_labels > 0]
            )
            del synthlayer

            # Create background:
            if backgnd_mode == "plain" or backgnd_mode == "rand":
                # convolve random noise with a 3d steerable spherical basis kernel
                # sampled from a normal distribution
                theta = np.random.uniform(0, 360)
                psi = np.random.uniform(0, 360)
                phi = np.random.uniform(-90, 90)
                spherical_basis_kernel = project_spherical_harmonic(
                    size=(7, 7, 7),
                    sigma=1.0,
                    l=2,
                    m=2,
                    angles=(theta, psi, phi),
                )
                x = np.random.randn(128, 128, 128)
                steerable_random_conv = convolve(x, spherical_basis_kernel)
                # alpha blend noise with foreground
                alpha = 3e-3
                synthimage = cv.addWeighted(
                    synthimage,
                    alpha,
                    steerable_random_conv.astype("float32"),
                    1 - alpha,
                    0,
                )

            elif backgnd_mode == "perlin":
                synthlayer = SampleConditionalGMM(idx_texture_labels)
                synthbackground = synthlayer(
                    [
                        tf.convert_to_tensor(
                            background_struct[np.newaxis, ..., np.newaxis],
                            dtype=tf.float32,
                        ),
                        tf.convert_to_tensor(
                            background_means[tf.newaxis, ..., tf.newaxis],
                            dtype=tf.float32,
                        ),
                        tf.convert_to_tensor(
                            background_stds[tf.newaxis, ..., tf.newaxis],
                            dtype=tf.float32,
                        ),
                    ]
                )[0, ..., 0].numpy()
                perlin_kernel = gaussian_3d_kernel(size=(7, 7, 7), sigma=1.0)
                # note that all angles are in degrees
                theta = np.random.uniform(0, 360)
                psi = np.random.uniform(0, 360)
                phi = np.random.uniform(-90, 90)
                spherical_basis_kernel = project_spherical_harmonic(
                    size=(7, 7, 7),
                    sigma=1.0,
                    l=2,
                    m=2,
                    angles=(theta, psi, phi),
                )
                x = np.random.randn(128, 128, 128)
                steerable_random_conv = convolve(x, spherical_basis_kernel).astype(
                    synthimage.dtype
                )  # typecast for alpha-blending
                alpha = 0.4
                # alpha blend perlin noise with foreground
                synthimage = cv.addWeighted(
                    synthimage, alpha, steerable_random_conv, 1 - alpha, 0
                )
                del synthlayer

            nib.save(
                nib.Nifti1Image(synthimage, affine=np.eye(4)),
                imgpath_base
                + "/{}/{}_{:04}.nii.gz".format(
                    os.path.basename(segs[i])[:-7],
                    os.path.basename(segs[i])[:-7],
                    j,
                ),
            )
            nib.save(
                nib.Nifti1Image(current_labels, affine=np.eye(4)),
                labpath_base
                + "/{}/{}_{:04}.nii.gz".format(
                    os.path.basename(segs[i])[:-7],
                    os.path.basename(segs[i])[:-7],
                    j,
                ),
            )
