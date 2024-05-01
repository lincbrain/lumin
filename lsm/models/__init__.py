def get_model(args):
    if args.model.framework == "cellpose":
        from .cellpose_wrapper import get_model
    else:
        raise NotImplementedError
    return get_model(args)
