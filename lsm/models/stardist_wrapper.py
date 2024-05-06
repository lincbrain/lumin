import cv2 as cv

import torch
import torch.nn as nn

from csbdeep.utils import normalize
from stardist.models import StarDist2D


class StarDist(nn.Module):
    def __init__(self, model_config: dict()):
        super(StarDist, self).__init__()

        self.model_config = model_config
        self.normalize = model_config["normalize"]

        model_type = model_config["model_type"]
        # TODO: add model_type assertion

        self.model = StarDist2D.from_pretrained(model_type)

    def forward(self, x: torch.Tensor):
        # reshape tensor
        x = x.squeeze(0).permute(1, 2, 0)
        x = x.detach().cpu().numpy()
        print(f"shape: {x.shape}")

        if self.normalize:
            x = normalize(x)

        masks, _ = self.model.predict_instances(x)

        return masks


def get_model(args):
    model_config = {
        "model_type": args.model.model_type,
        "normalize": args.model.normalize,
    }

    model = StarDist(model_config=model_config)

    return model


if __name__ == "__main__":
    model = StarDist2D.from_pretrained("2D_versatile_he")
    img_path = (
        "/om2/user/ckapoor/lsm-data/NeurIPS22-CellSeg/Training/images/cell_00902.png"
    )
    img = cv.imread(img_path)
    labels, _ = model.predict_instances(normalize(img))
    import numpy as np

    color_map = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]
    import cv2 as cv

    seg_mask = color_map[labels]
    overlay = cv.addWeighted(img, 0.7, seg_mask, 0.3, 0)
    cv.imwrite(f"stardist_orig.png", img)
    cv.imwrite(f"mask.png", seg_mask)
    # cv.imwrite(f"stardist_overlay_mask.png", overlay)
    print(f"hello")
