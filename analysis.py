import os
import sys
import time
import itertools
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
from lsm.evaluation.segmentation_metrics import SegmentationMetrics


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

    # load some universal info
    models = args.analysis.models

    # load data
    data_dir = args.data.data_dir
    chunks = args.analysis.stitching.chunk_sizes
    # TODO: change gt + stitched volume loading
    # gt_vol = imread(os.path.join(exp_dir, "gt_proxy.tiff"))
    # stitched_vols = [
    #    imread(os.path.join(exp_dir, f"chunk_{csize}.tiff")) for csize in chunks
    # ]

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

    elif args.analysis.do_segmentation:
        # all possible model pairs
        model_pairs = list(itertools.combinations(models, 2))
        metrics = args.analysis.segmentation.metrics

        # load corresponding volumes
        for pair in model_pairs:
            print(f"evaluating model pair: {pair}")
            model1_path = os.path.join(exp_dir, f"{pair[0]}_seg")
            model2_path = os.path.join(exp_dir, f"{pair[1]}_seg")

            imnames = sorted(os.listdir(model1_path))

            for imname in imnames:
                model1_out = imread(os.path.join(model1_path, imname))
                model2_out = imread(os.path.join(model2_path, imname))

                # instantiate metric class
                seg_metrics = SegmentationMetrics(
                    metric=None,
                    vol1=model1_out,
                    vol2=model2_out,
                )

                # compute segmentation metrics for each model pair
                for metric in metrics:
                    seg_metrics.metric = metric

                    if args.analysis.segmentation.analyse_prob:
                        print(f"loading model output probability maps")
                        # load probability map outputs
                        model1_prob = imread(
                            os.path.join(model1_path, f"probmap_{pair[0]}.tiff")
                        )
                        model2_prob = imread(
                            os.path.join(model2_path, f"probmap_{pair[1]}.tiff")
                        )
                        seg_metrics.prob1 = model1_prob
                        seg_metrics.prob2 = model2_prob

                    # compute relevant metric for the model pair
                    # and construct relevant plots
                    computed_metric = seg_metrics.compute_metric()
                    seg_metrics.plot_metric()

    else:
        raise ValueError("perform either segmentation or stitching analysis")


if __name__ == "__main__":
    parser = create_args_parser()
    parser.add_argument("--ddp", action="store_true", help="Distributed processing")
    args, unknown = parser.parse_known_args()
    config = load_config(args, unknown)
    main_function(config)
