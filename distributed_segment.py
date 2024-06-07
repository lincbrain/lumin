import os
import sys
import time
import numpy as np
from tqdm import tqdm

from tifffile import imwrite

import dask
from dask.diagnostics import ProgressBar

from lsm.dataio import get_data
from lsm.utils.logger import Logger
from lsm.distributed import get_model
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


def main_function(args):
    init_env(args)
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    exp_dir = args.training.exp_dir

    logger = Logger(
        log_dir=exp_dir,
        save_dir=os.path.join(exp_dir, "chunks"),
        monitoring=args.training.get("monitoring", "tensorboard"),
        monitoring_dir=os.path.join(exp_dir, "events"),
        rank=rank,
        is_master=is_master(),
        multi_process_logging=(world_size > 1),
    )

    log.info(f"Segmentation directory: {exp_dir}")

    # lazy load data as a dask array
    dataset = get_data(args)

    # TODO: save GT image volume too
    gt_path = os.path.join(exp_dir, f"gt_proxy.tiff")
    gt_vol = next(iter(dataset))[-1]["orig_vol"]
    imwrite(gt_path, gt_vol)

    # run distributed segmentation
    for model in tqdm(args.segmentation.models):
        print(f"Running stitching with {model}...")
        # load model, as a segmentation function
        segment_func = get_model(model=model)
        if model == "cellpose":
            img_vol = next(iter(dataset))[-1]
            cfg_dict = {
                "image": img_vol["orig_vol"],
                "debug": args.model.debug,
                "channels": args.model.channels,
                "boundary": args.model.boundary,
                "diameter": args.model.diameter,
                "use_anisotropy": args.model.use_anisotropy,
                "iou_depth": args.model.stitching.iou_depth,
                "iou_threshold": args.model.stitching.iou_threshold,
            }

        elif model in ["anystar", "anystar-gaussian"]:
            return None
        elif model == "stardist3d":
            return None
        else:
            raise NotImplementedError

        for chunk in tqdm(args.segmentation.chunk_sizes):
            print(f"Running segmentation for chunk size: {chunk}")
            cfg_dict["chunk"] = chunk
            seg_vol = segment_func(**cfg_dict)

            with ProgressBar():
                with dask.config.set(scheduler="synchronous"):
                    seg_vol = seg_vol.compute()
                    fpath = os.path.join(exp_dir, f"chunk_{chunk}.tiff")
                    imwrite(fpath, seg_vol)


if __name__ == "__main__":
    parser = create_args_parser()
    parser.add_argument("--ddp", action="store_true", help="Distributed processing")
    args, unknown = parser.parse_known_args()
    config = load_config(args, unknown)
    main_function(config)
