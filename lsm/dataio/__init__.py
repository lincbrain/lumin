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
    elif dataset_type == "dandiset":
        # add extra parameters
        # data
        cfgs["url"] = args.data.url
        cfgs["scale"] = args.data.scale
        # voxel
        cfgs["vol_lim"] = args.segmentation.vol_lims
        cfgs["voxel_shapes"] = args.segmentation.voxel_shape

        from .lsm_dandiset import ImageDataset
    else:
        raise NotImplementedError

    cfgs.update(overwrite_cfgs)
    dataset = ImageDataset(**cfgs)
    if return_val:
        val_dataset = ImageDataset(**cfgs)
        return dataset, val_dataset
    else:
        return dataset
