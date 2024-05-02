import os
import torch
import pickle
import imageio
import torchvision
import cv2 as cv
import numpy as np

import torch.distributed as dist

from lsm.utils.load_config import cond_mkdir

color_map = np.random.randint(0, 256, (500, 3), dtype=np.uint8)
color_map[0] = [0, 0, 0]


class Logger(object):
    def __init__(
        self,
        log_dir,
        save_dir,
        monitoring=None,
        monitoring_dir=None,
        rank=0,
        is_master=True,
        multi_process_logging=True,
    ):
        self.stats = dict()
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.rank = rank
        self.is_master = is_master
        self.multi_process_logging = multi_process_logging

        if self.is_master:
            cond_mkdir(self.log_dir)
        if self.multi_process_logging:
            dist.barrier()

        self.monitoring = None
        self.monitoring_dir = None

        if not (monitoring is None or monitoring == "none"):
            self.setup_monitoring(monitoring, monitoring_dir)

    def setup_monitoring(self, monitoring, monitoring_dir):
        self.monitoring = monitoring
        self.monitoring_dir = monitoring_dir
        if monitoring == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(self.monitoring_dir)
        else:
            raise NotImplementedError(
                'Monitoring tool "%s" not supported!' % monitoring
            )

    def add(self, category, k, v, it):
        if category not in self.stats:
            self.stats[category] = {}

        if k not in self.stats[category]:
            self.stats[category][k] = []

        self.stats[category][k].append((it, v))

        k_name = "%s/%s" % (category, k)
        if self.monitoring == "telemetry":
            self.tm.metric_push_async({"metric": k_name, "value": v, "it": it})
        elif self.monitoring == "tensorboard":
            self.tb.add_scalar(k_name, v, it)

    def add_imgs_eval(self, imgs, class_name, it, save_seg=False):
        outdir = os.path.join(self.save_dir, class_name)
        if self.is_master and not os.path.exists(outdir):
            os.makedirs(outdir)
        if self.multi_process_logging:
            dist.barrier()
        outfile = os.path.join(outdir, "{:08d}_{}.png".format(it, self.rank))

        # color map segmentation masks, if applicable
        if save_seg:
            imgs = color_map[imgs]
        else:
            imgs = imgs.squeeze(0).detach().cpu().numpy()

        cv.imwrite(outfile, imgs)

        # potential TODO: log to tensorboard

    def add_imgs(self, imgs, class_name, it):
        outdir = os.path.join(self.save_dir, class_name)
        if self.is_master and not os.path.exists(outdir):
            os.makedirs(outdir)
        if self.multi_process_logging:
            dist.barrier()
        outfile = os.path.join(outdir, "{:08d}_{}.png".format(it, self.rank))

        imgs = torchvision.utils.make_grid(imgs)
        torchvision.utils.save_image(imgs.clone(), outfile, nrow=8)

        if self.monitoring == "tensorboard":
            self.tb.add_image(class_name, imgs, global_step=it)

    def add_figure(self, fig, class_name, it, save_img=True):
        if save_img:
            outdir = os.path.join(self.save_dir, class_name)
            if self.is_master and not os.path.exists(outdir):
                os.makedirs(outdir)
            if self.multi_process_logging:
                dist.barrier()
            outfile = os.path.join(outdir, "{:08d}_{}.png".format(it, self.rank))

            image_hwc = io_util.figure_to_image(fig)
            imageio.imwrite(outfile, image_hwc)
            if self.monitoring == "tensorboard":
                if len(image_hwc.shape) == 3:
                    image_hwc = np.array(image_hwc[None, ...])
                self.tb.add_images(
                    class_name,
                    torch.from_numpy(image_hwc),
                    dataformats="NHWC",
                    global_step=it,
                )
        else:
            if self.monitoring == "tensorboard":
                self.tb.add_figure(class_name, fig, it)

    def add_module_param(self, module_name, module, it):
        if self.monitoring == "tensorboard":
            for name, param in module.named_parameters():
                self.tb.add_histogram(
                    "{}/{}".format(module_name, name), param.detach(), it
                )

    def get_last(self, category, k, default=0.0):
        if category not in self.stats:
            return default
        elif k not in self.stats[category]:
            return default
        else:
            return self.stats[category][k][-1][1]

    def save_stats(self, filename):
        filename = os.path.join(self.log_dir, filename + "_{}".format(self.rank))
        with open(filename, "wb") as f:
            pickle.dump(self.stats, f)

    def load_stats(self, filename):
        filename = os.path.join(self.log_dir, filename + "_{}".format(self.rank))
        if not os.path.exists(filename):
            return

        try:
            with open(filename, "rb") as f:
                self.stats = pickle.load(f)
                log.info(f"Load file: {filename}")
        except EOFError:
            log.info("Warning: log file corrupted!")
