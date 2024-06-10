def get_model(model):
    if model == "cellpose":
        from .segment_cellpose import segment
    elif model in ["anystar", "anystar-gaussian", "anystar-spherical"]:
        from .segment_steerable import segment
    elif model == "stardist3d":
        from .segment_cellpose import segment
    else:
        raise NotImplementedError(
            f"{model} not implemented, choose one of [cellpose, anystar, anystar-gaussian. stardist3d]"
        )

    return segment
