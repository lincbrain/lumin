from typing import Optional

import torch
import torch.nn as nn

from cellpose import models, io


class Cellpose(nn.Module):
    def __init__(
        self,
        model_config: dict(),
        device: Optional[torch.device] = torch.device("cuda:0"),
    ):
        super(Cellpose, self).__init__()

        model_type = model_config["model_type"]
        self.channels = model_config["channels"]
        self.n_channels = model_config["n_channels"]
        self.model = models.Cellpose(
            gpu=True, nchan=self.n_channels, model_type=model_type
        )

    def forward(self, x: torch.Tensor):
        masks, flows, styles, diams = self.model.eval(
            x, diameter=None, channels=self.channels
        )

        return masks


def get_model(args):
    model_config = {
        "model_type": args.model.model_type,
        "channels": args.model.channels,
        "n_channels": args.model.n_channels,
    }

    model = Cellpose(model_config=model_config)

    return model


if __name__ == "__main__":
    model = Cellpose(
        model_config={
            "model_type": "neurips_cellpose_default",
            "channels": None,
            "n_channels": 3,
        },
        device=torch.device("cuda:0"),
    )
    # test with 3-channel image batch
    from skimage.io import imread

    img_path = (
        "/om2/user/ckapoor/lsm-data/NeurIPS22-CellSeg/Training/images/cell_00902.png"
    )
    img = torch.from_numpy(imread(img_path))
    out = model(img)
    import numpy as np

    print(f"uq: {np.unique(out)}")
    import cv2 as cv

    cv.imwrite("/om2/user/ckapoor/lsm-segmentation/test_00902.png", out)
    print(f"out: {out.shape}")
