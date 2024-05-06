from typing import Optional

import torch
import torch.nn as nn


class CellProfiler(nn.Module):
    def __init__(self, model_config: dict()):
        return None

    def forward(self, x: torch.Tensor):
        return None


def get_model(args):
    model_config = {}

    model = CellProfiler(model_config=model_config)

    return model


if __name__ == "__main__":
    print(f"hello")
