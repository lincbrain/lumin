def get_data(args, return_val=False, val_downscale=1.0, **overwrite_cfgs):
    dataset_type = args.data.dataset_type
    # dataset_type = args.data.get("type", "neurips_challenge")
    cfgs = {
        "data_dir": args.data.data_dir,
    }

    if dataset_type == "neurips_challenge":
        from .neurips_challenge import ImageDataset
    elif dataset_type == "anystar_nii":
        from .anystar_nii import ImageDataset
    else:
        raise NotImplementedError

    cfgs.update(overwrite_cfgs)
    dataset = ImageDataset(**cfgs)
    if return_val:
        # cfgs["downscale"] = val_downscale
        val_dataset = ImageDataset(**cfgs)
        return dataset, val_dataset
    else:
        return dataset
