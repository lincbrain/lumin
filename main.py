import os
import sys
import time
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader

from lsm.dataio import get_data
from lsm.models import get_model
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


def main_function(args):
    init_env(args)
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    i_backup = (
        int(args.training.i_backup // world_size) if args.training.i_backup > 0 else -1
    )
    i_val = int(args.training.i_val // world_size) if args.training.i_val > 0 else -1
    exp_dir = args.training.exp_dir

    device = torch.device("cuda", local_rank)

    logger = Logger(
        log_dir=exp_dir,
        save_dir=os.path.join(exp_dir, "predictions"),
        monitoring=args.training.get("monitoring", "tensorboard"),
        monitoring_dir=os.path.join(exp_dir, "events"),
        rank=rank,
        is_master=is_master(),
        multi_process_logging=(world_size > 1),
    )

    log.info(f"Experiments directory: {exp_dir}")

    if is_master():
        pass

    # get data from dataloader
    dataset, val_dataset = get_data(args=args, return_val=True)

    batch_size = args.data.get("batch_size", None)

    if args.ddp:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset, sampler=train_sampler, batch_size=batch_size
        )
        val_sampler = DistributedSampler(val_dataset)
        valloader = torch.utils.data.DataLoader(
            val_dataset, sampler=val_sampler, batch_size=batch_size
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=args.data.get("pin_memory", False),
        )
        valloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # create model
    model = get_model(args)
    model.to(device)
    log.info(model)
    model_name = args.model.framework

    if world_size > 1:
        dist.barrier()

    tick = time.time()
    log.info(f"Start evaluating in {exp_dir}")

    it, epoch_idx = 0, 0
    end = it >= len(dataset)

    with tqdm(range(len(dataset)), disable=not is_master()) as pbar:
        if is_master():
            pbar.update()

        while it <= len(dataset) and not end:
            try:
                if args.ddp:
                    raise NotImplementedError
                for (indices, model_input, ground_truth) in dataloader:
                    int_it = int(it // world_size)

                    # TODO: add predicted images
                    _, rgb, labels = next(iter(valloader))

                    gt_rgb = rgb["rgb"]
                    gt_mask = labels["mask"]

                    import numpy as np

                    predicted_masks = torch.from_numpy(model(gt_rgb).astype(np.uint8))

                    print(f"gt rgb: {gt_rgb.dtype}")
                    print(f"gt mask: {gt_mask.dtype}")
                    print(f"pmask: {predicted_masks.dtype}")

                    # logger.add_imgs(
                    #    imgs=gt_rgb,
                    #    class_name="gt_rgb",
                    #    it=it,
                    # )
                    # logger.add_imgs(
                    #    imgs=gt_mask,
                    #    class_name="gt_mask",
                    #    it=it,
                    # )
                    logger.add_imgs(
                        imgs=predicted_masks,
                        class_name="predicted_mask",
                        it=it,
                    )

                    if it >= len(dataset):
                        end = True
                        break

                    start_time = time.time()

                    end_time = time.time()
                    log.debug(
                        "One iteration time is {:.2f}".format(end_time - start_time)
                    )

                    it += world_size
                    if is_master():
                        pbar.update(world_size)

                epoch_idx += 1

            except KeyboardInterrupt:
                if is_master():
                    print(f"TODO: idk")
                sys.exit()


if __name__ == "__main__":
    parser = create_args_parser()
    parser.add_argument("--ddp", action="store_true", help="Distributed processing")
    args, unknown = parser.parse_known_args()
    config = load_config(args, unknown)
    main_function(config)
