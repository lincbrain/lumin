import torch
import torch.nn as nn

from cellpose import models, io


class Cellpose(nn.Module):
    def __init__(self, model_config: dict()):
        super(Cellpose, self).__init__()

        model_type = model_config["model_type"]
        self.channels = model_config["channels"]
        self.model = models.Cellpose(model_type=model_type)

    def forward(self, x: torch.Tensor):
        masks, flows, styles, diams = self.model.eval(
            x, diameter=None, channels=self.channels
        )

        return masks


def get_model(args):
    model_config = {
        "model_type": args.model.model_type,
        "channels": args.model.channels,
    }

    model = Cellpose(model_config=model_config)

    return model


if __name__ == "__main__":
    model = Cellpose(model_config={"model_type": "cyto3"})
    # test with 3-channel image batch
    img = torch.randn(4, 3, 256, 256)
    out = model(img)
    print(f"out: {out.shape}")
