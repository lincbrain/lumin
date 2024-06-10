import os
import sys
import time
import numpy as np
from tqdm import tqdm

from tifffile import imread

import torch

from lsm.utils.logger import Logger
from lsm.utils.console_log import log
from lsm.utils.train_utils import count_trainable_parameters
from lsm.utils.load_config import create_args_parser, load_config, backup
from lsm.utils.distributed_util import (
    init_env,
    get_rank,
    is_master,
    get_local_rank,
    get_world_size,
)

from lsm.evaluation.stitch_metrics import StitchMetrics
from lsm.evaluation.stitching_stats import StitchingAnalysis


def main_function(args):
    init_env(args)
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    exp_dir = args.training.exp_dir

    logger = Logger(
        log_dir=exp_dir,
        save_dir=os.path.join(exp_dir, "plots"),
        monitoring=args.training.get("monitoring", "tensorboard"),
        monitoring_dir=os.path.join(exp_dir, "events"),
        rank=rank,
        is_master=is_master(),
        multi_process_logging=(world_size > 1),
    )

    log.info(f"Analysis log directory: {exp_dir}")

    # load data
    data_dir = args.data.data_dir
    chunks = args.analysis.stitching.chunk_sizes
    gt_vol = imread(os.path.join(exp_dir, "gt_proxy.tiff"))
    stitched_vols = [
        imread(os.path.join(exp_dir, f"chunk_{csize}.tiff")) for csize in chunks
    ]

    # plot iou + nuclei count
    if args.analysis.do_stitching:
        # compute stitching metrics
        for metric in args.analysis.stitching.metrics:
            print(f"Computing {metric} for stitching...")
            stitching_metrics = StitchMetrics(
                gt_proxy=gt_vol,
                stitched_vols=stitched_vols,
                metric=metric,
                chunk_sizes=chunks,
            )
            metric_dict = stitching_metrics.compute_metric()
            # create and save plots
            stitching_stats = StitchingAnalysis(
                metric_dict=metric_dict,
                metric_name=metric,
                var_name="Chunk Size",
                save_path=exp_dir,
                resolution=args.analysis.resolution,
            )
            all_stats, kruskal_result = stitching_stats.all_analysis()
            print(f"Descriptive statistics: {all_stats}")
            print(f"Results from kruskal's h-test (p value): {kruskal_result:.3f}")


if __name__ == "__main__":
    parser = create_args_parser()
    parser.add_argument("--ddp", action="store_true", help="Distributed processing")
    args, unknown = parser.parse_known_args()
    config = load_config(args, unknown)
    main_function(config)
