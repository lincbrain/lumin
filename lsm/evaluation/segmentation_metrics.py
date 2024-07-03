import os
import numpy as np
import pandas as pd
from typing import Optional

from scipy.stats import entropy
from scipy.special import kl_div
from scipy.spatial import distance
from skimage.metrics import hausdorff_distance
from skimage.measure import regionprops, regionprops_table, label


class SegmentationMetrics:
    # pairwise, corresponding model metric comparison
    # for a range of metrics (geometric, spatial and probabilistic)
    def __init__(
        self,
        metric: str,
        vol1: np.ndarray,
        vol2: np.ndarray,
        prob1: Optional[np.ndarray] = None,
        prob2: Optional[np.ndarray] = None,
    ):
        self.vol1 = vol1
        self.vol2 = vol2
        self.eps = 1e-9  # constant for numerical stability
        self.metric = metric

        if prob1 is not None and prob2 is not None:
            self.prob1 = prob1
            self.prob2 = prob2

        assert (
            self.vol1.shape == self.vol2.shape
        ), "Volumes must have same shape for metric comparison"

        # relabel volume, find instance mappings
        (
            nearest_neighbors,
            self.label1,
            self.label2,
            self.coords1,
            self.coords2,
        ) = self._nearest_neighbors(vol1=self.vol1, vol2=self.vol2)
        self.instance_mapping = self._corresponding_instance_map(
            pairwise_dist=nearest_neighbors,
            label_vol1=self.label1,
            label_vol2=self.label2,
        )

    def _binarize_vol(self, vol: np.ndarray):
        # binarize volume to be in [0, 255]
        binary_vol = np.zeros_like(vol, dtype=vol.dtype)
        binary_vol[vol != 0] = 255
        return binary_vol

    def _compute_centroids_and_coords(self, vol: np.ndarray):
        # compute centroids of instance labels
        binary_vol = self._binarize_vol(vol=vol)
        label_vol = label(binary_vol)  # consistent instance relabeling
        segmentation_props = regionprops(label_vol)
        # return a vector of centroids and label coordinates
        centroids = [prop.centroid for prop in segmentation_props]
        coords = [prop.coords for prop in segmentation_props]
        return centroids, label_vol, coords

    def _nearest_neighbors(self, vol1: np.ndarray, vol2: np.ndarray):
        # compute the 1st nearest neighbor between 2 volumes
        # we use this as a heuristic to determine corresponding nuclei
        centroids1, label_vol1, coords1 = self._compute_centroids_and_coords(vol=vol1)
        centroids2, label_vol2, coords2 = self._compute_centroids_and_coords(vol=vol2)

        # check for empty masks
        if not centroids1 or not centroids2:
            raise ValueError("One of the masks has no nuclei instances")

        # compute pairwise distances
        distances = distance.cdist(centroids1, centroids2, metric="euclidean")

        # compute nearest neighbors
        nearest_neighbors = np.argmin(distances, axis=1)

        return nearest_neighbors, label_vol1, label_vol2, coords1, coords2

    def _corresponding_instance_map(
        self, pairwise_dist: np.ndarray, label_vol1: np.ndarray, label_vol2: np.ndarray
    ):
        # compute corresponding instance labels
        instance_mapping = {}

        for idx, nearest_neighbor in enumerate(pairwise_dist):
            instance_label1 = label_vol1[label_vol1 == idx + 1][0]
            instance_label2 = label_vol2[label_vol2 == nearest_neighbor + 1][0]
            # assign instance mapping
            instance_mapping[instance_label1] = instance_label2

        return instance_mapping

    def haussdorf_distance(self):
        # compute (directed) hausdorff distance between corresponding
        # instance masks
        hausdorff_distances = {}
        # binarize volumes first
        binary_vol1, binary_vol2 = self._binarize_vol(
            vol=self.vol1
        ), self._binarize_vol(vol=self.vol2)

        for instance_label1, instance_label2 in self.instance_mapping.items():
            # coordinates of points in each instance
            c1 = self.coords1[instance_label1 - 1]
            c2 = self.coords2[instance_label2 - 1]

            # compute hausdorff distance
            hausdorff = max(
                distance.directed_hausdorff(c1, c2)[0],
                distance.directed_hausdorff(c2, c1)[0],
            )
            hausdorff_distances[instance_label1] = hausdorff

        return hausdorff_distances

    def mean_volume_and_axis(self, vol: np.ndarray):
        # compute nuclei volumes
        segmentation_props = regionprops(vol)
        props = regionprops_table(
            vol,
            properties=(
                "area",
                "axis_major_length",
                # "axis_minor_length",
            ),
        )
        props_table = pd.DataFrame(props)

        return props_table

    def kl_divergence(self):
        # compute KL divergence between probability maps
        return kl_div(self.prob1, self.prob2)

    def entropy(self, prob_map: np.ndarray):
        # compute entropy for a probability map
        # note: using base 2 here does computations in the unit of bits
        return entropy(prob_map, base=2)

    def nuclei_mse(self):
        # compute MSE between geometric centers of nuclei
        centroids1, _, _ = self._compute_centroids_and_coords(vol=self.vol1)
        centroids2, _, _ = self._compute_centroids_and_coords(vol=self.vol2)

        distances = []
        for instance_label1, instance_label2 in self.instance_mapping.items():
            c1 = np.array(centroids1[instance_label1 - 1])
            c2 = np.array(centroids2[instance_label2 - 1])
            dist = np.linalg.norm(c1 - c2)
            distances.append(dist)

        return distances

    def jaccard_index(self):
        # compute jaccard metric
        intersection = np.logical_and(self.vol1, self.vol2)
        union = np.logical_or(self.vol1, self.vol2)

        return intersection.sum() / (float(union.sum()) + self.eps)

    def plot_metric(self):
        # relevant plots
        return None

    def compute_metric(self):
        metric_list = []

        if self.metric == "hausdorff":
            hausdorff = self.haussdorf_distance()
            self.metric_val = hausdorff
            metric_list.append(hausdorff)

        elif self.metric == "mean_geometry":
            print(f"computing mean geometry metrics...")
            tab1 = self.mean_volume_and_axis(vol=self.vol1)
            tab2 = self.mean_volume_and_axis(vol=self.vol2)
            metric_list.append(tab1)
            metric_list.append(tab2)

        elif self.metric == "nuclei_distance":
            print(f"computing corresponding nuclei distance...")
            distances = self.nuclei_mse()
            metric_list.append(distances)

        elif self.metric == "kl_divergence":
            print(f"computing KL-divergence...")
            kld = self.kl_divergence()
            mean_kld = np.mean(kld)
            metric_list.append(mean_kld)

        elif self.metric == "entropy":
            print(f"computing entropy...")
            e1 = self.entropy(prob_map=self.prob1)
            e2 = self.entropy(prob_map=self.prob2)
            me1, me2 = np.mean(e1), np.mean(e2)
            metric_list.append(me1)
            metric_list.append(me2)

        else:
            raise NotImplementedError(
                f"{self.metric} not implemented, choose one of [hausdorff, mean_geometry, nuclei_distance, kl_divergence, entropy]"
            )

        return metric_list


if __name__ == "__main__":
    from tifffile import imread

    m1 = imread(
        "/om2/user/ckapoor/lsm-segmentation/model_analysis/stitching/anystar_seg/chunk_64.tiff"
    )
    m2 = imread(
        "/om2/user/ckapoor/lsm-segmentation/model_analysis/stitching/anystar-gaussian_seg/chunk_64.tiff"
    )
    p1 = imread("/om2/user/ckapoor/lsm-segmentation/probmap_anystar.tiff")
    p2 = imread("/om2/user/ckapoor/lsm-segmentation/probmap_gaussian.tiff")
    metrics = [
        "hausdorff",
        "mean_geometry",
        "kl_divergence",
        "entropy",
        "nuclei_distance",
    ]
    for metric in metrics:
        seg_metrics = SegmentationMetrics(
            metric=metric, vol1=m1, vol2=m2, prob1=p1, prob2=p2
        )
        m = seg_metrics.compute_metric()
        print(m)
