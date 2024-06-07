import numpy as np
from typing import List


class StitchMetrics:
    def __init__(
        self,
        gt_proxy: np.ndarray,
        stitched_vols: List[np.ndarray],
        metric: str,
        chunk_sizes: List[int],
    ):
        self.gt = gt_proxy  # we don't really have the gt, just a proxy for it
        self.stitched_vols = stitched_vols
        self.metric = metric
        self.chunk_sizes = chunk_sizes

        # define a constant for numerical stability
        self.eps = 1e-7

    def _binarize_vol(self, vol: np.ndarray):
        # binarize an image to [0, 255]
        binary_vol = np.zeros_like(vol, dtype=vol.dtype)
        binary_vol[vol != 0] = 255

        return binary_vol

    def compute_mean_iou(self, vol: np.ndarray):
        # we modify the volumes to be binarized
        # binarize volumes
        if self.gt.shape != vol.shape:
            raise ValueError(f"volumes must have identical shapes")

        # binarize volumes
        binary_gt = self._binarize_vol(vol=self.gt)
        binary_seg = self._binarize_vol(vol=vol)

        # compute iou
        intersection = np.logical_and(binary_gt, binary_seg).sum()
        union = np.logical_or(binary_gt, binary_seg).sum()
        iou = intersection / (union + self.eps)

        return iou

    def nuclei_count(self, vol: np.ndarray):
        return len(np.unique(vol))

    def compute_metric(self):
        metric_dict = {}

        if "iou" in self.metric:
            metric_func = self.compute_mean_iou
        elif "count" in self.metric:
            metric_func = self.nuclei_count
            metric_dict["GT"] = metric_func(vol=self.gt)
        else:
            raise NotImplementedError(
                f"{self.metric} not implemented, choose one of [iou, count]"
            )

        for idx, chunk in enumerate(self.chunk_sizes):
            metric_dict[str(chunk)] = metric_func(vol=self.stitched_vols[idx])

        return metric_dict


if __name__ == "__main__":
    gt = np.random.randn(64, 64, 64)
    test = np.random.randn(64, 64, 64)
    vols = [test, test, test]
    metric_class = StitchMetrics(
        gt_proxy=gt, stitched_vols=vols, metric="iou", chunk_sizes=[20, 30]
    )

    d = metric_class.compute_metric()
    print(f"metric dict: {d}")
