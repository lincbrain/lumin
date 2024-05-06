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

        self.model_config = model_config
        model_type = model_config["model_type"]
        self.channels = model_config["channels"]
        self.n_channels = int(model_config["n_channels"])
        self.model = models.Cellpose(
            gpu=True, nchan=self.n_channels, model_type=model_type
        )

    def forward(self, x: torch.Tensor):
        masks, flows, styles, diams = self.model.eval(
            x,
            diameter=self.model_config["diameters"],
            channels=self.channels,
            normalize=self.model_config["normalize"],
            tile_overlap=self.model_config["tile_overlap"],
            augment=self.model_config["augment"],
        )

        return masks


def get_model(args):
    model_config = {
        "model_type": args.model.model_type,
        "channels": args.model.channels,
        "n_channels": args.model.n_channels,
        "diameters": args.model.diameters,
        "normalize": args.model.normalize,
        "tile_overlap": args.model.tile_overlap,
        "augment": args.model.augment,
    }

    model = Cellpose(model_config=model_config)

    return model


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = Cellpose(
        model_config={
            "model_type": "nuclei",
            "channels": [[0, 0]],
            "n_channels": 2,
            "diameters": 15,
            "normalize": True,
            "tile_overlap": 0.1,
            "augment": True,
        },
        device=torch.device("cuda:0"),
    )
    import cv2 as cv
    import numpy as np

    img_path = f"/om2/user/ckapoor/lsm-segmentation/test-lsm.png"
    img = cv.imread(img_path)

    out = model(img)
    color_map = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]
    seg_mask = color_map[out]
    overlay = cv.addWeighted(img, 0.7, seg_mask, 0.3, 0)
    cv.imwrite(f"overlay_lsm_seg.png", overlay)

    model = Cellpose(
        model_config={
            "model_type": "neurips_cellpose_default",
            "channels": None,
            "n_channels": 3,
            "diameters": None,
            "normalize": False,
            "tile_overlap": 0.6,
            "augment": True,
        },
        device=torch.device("cuda:0"),
    )
    # test with 3-channel image batch
    import cv2 as cv
    import numpy as np
    from cellpose import io, transforms

    img_path = (
        "/om2/user/ckapoor/lsm-data/NeurIPS22-CellSeg/Training/images/cell_00902.png"
    )
    img = io.imread(img_path)
    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    img_norm = transforms.normalize_img(img, axis=-1)
    out = model(img_norm)

    color_map = np.random.randint(0, 256, (500, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # ensure black bg
    seg_mask = color_map[out]
    cv.imwrite(f"test_seg.png", seg_mask)
    cv.imwrite(f"test_gt_img.png", img)

    print(f"uq: {np.unique(out)}")
