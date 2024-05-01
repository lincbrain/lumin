import os
import copy
import yaml
import addict
import shutil
import argparse

import torch

from lsm.utils.console_log import log


class ForceKeyErrorDict(addict.Dict):
    # excpetion class for missing keys in config files
    def __missing__(self, name):
        raise KeyError(name)


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def backup(backup_dir):
    """automatic backup"""
    log.info("Backing up... ")
    special_files_to_copy = []
    filetypes_to_copy = [".py"]
    subdirs_to_copy = ["", "lsm"]

    this_dir = "./"  # TODO
    cond_mkdir(backup_dir)
    # special files
    [
        cond_mkdir(os.path.join(backup_dir, os.path.split(file)[0]))
        for file in special_files_to_copy
    ]
    [
        shutil.copyfile(os.path.join(this_dir, file), os.path.join(backup_dir, file))
        for file in special_files_to_copy
    ]
    # dirs
    for subdir in subdirs_to_copy:
        cond_mkdir(os.path.join(backup_dir, subdir))
        files = os.listdir(os.path.join(this_dir, subdir))
        files = [
            file
            for file in files
            if os.path.isfile(os.path.join(this_dir, subdir, file))
            and file[file.rfind(".") :] in filetypes_to_copy
        ]
        [
            shutil.copyfile(
                os.path.join(this_dir, subdir, file),
                os.path.join(backup_dir, subdir, file),
            )
            for file in files
        ]
    log.info("Done!")


def load_yaml(path: str, default_path=None):
    """load YAML file from a specified path"""
    with open(path, encoding="utf8") as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = ForceKeyErrorDict(**config_dict)

    if default_path is not None and path != default_path:
        with open(default_path, encoding="utf8") as default_yaml_file:
            default_config_dict = yaml.load(default_yaml_file, Loader=yaml.FullLoader)
            main_config = ForceKeyErrorDict(**default_config_dict)

        main_config.update(config)
        config = main_config

    return config


def save_config(datadict: ForceKeyErrorDict, path: str):
    """save config to a specified path"""
    datadict = copy.deepcopy(datadict)
    datadict.training.ckpt_file = None
    datadict.training.pop("exp_dir")
    with open(path, "w", encoding="utf8") as outfile:
        yaml.dump(datadict.to_dict(), outfile, default_flow_style=False)


def update_config(config, unknown):
    """update config given args"""
    for idx, arg in enumerate(unknown):
        if arg.startswith("--"):
            if (":") in arg:
                k1, k2 = arg.replace("--", "").split(":")
                argtype = type(config[k1][k2])
                if argtype == bool:
                    v = unknown[idx + 1].lower() == "true"
                else:
                    if config[k1][k2] is not None:
                        v = type(config[k1][k2])(unknown[idx + 1])
                    else:
                        v = unknown[idx + 1]
                print(f"Changing {k1}:{k2} ---- {config[k1][k2]} to {v}")
                config[k1][k2] = v
            else:
                k = arg.replace("--", "")
                v = unknown[idx + 1]
                argtype = type(config[k])
                print(f"Changing {k} ---- {config[k]} to {v}")
                config[k] = v

    return config


def create_args_parser():
    """create a argument parser object"""
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument(
        "--resume_dir", type=str, default=None, help="Directory of experiment to load."
    )
    return parser


def load_config(args, unknown, base_config_path=None):
    """master function to load and return a config from command line arguments"""
    assert (args.config is not None) != (
        args.resume_dir is not None
    ), "you must specify ONLY one in 'config' or 'resume_dir' "

    found_k = None
    for item in unknown:
        if "local_rank" in item:
            found_k = item
            break
    if found_k is not None:
        unknown.remove(found_k)

    print("Parse extra configs: ", unknown)

    if args.resume_dir is not None:
        assert (
            args.config is None
        ), "given --config will not be used when given --resume_dir"
        assert (
            "--expname" not in unknown
        ), "given --expname with --resume_dir will lead to unexpected behavior."
        # if loading from a directory, do not use base.yaml as the default;
        config_path = os.path.join(args.resume_dir, "config.yaml")
        config = load_yaml(config_path, default_path=None)

        # use configs given by command line to further overwrite current config
        config = update_config(config, unknown)

        # use the loading directory as the experiment path
        config.training.exp_dir = args.resume_dir
        print(f"=> Loading previous experiments in: {config.training.exp_dir}")
    else:
        # use base.yaml as default when loading from a config file
        config = load_yaml(args.config, default_path=base_config_path)

        # use command line configs to overwrite default
        config = update_config(config, unknown)

        # use expname and log_root_dir to get experiment directory
        if "exp_dir" not in config.training:
            config.training.exp_dir = os.path.join(
                config.training.log_root_dir, config.expname
            )

    # add other configs in args to config
    other_dict = vars(args)
    other_dict.pop("config")
    other_dict.pop("resume_dir")
    config.update(other_dict)

    if hasattr(args, "ddp") and args.ddp:
        if config.device_ids != -1:
            print("Ignoring device_ids configs when using DDP. Auto set to -1.")
            config.device_ids = -1
    else:
        args.ddp = False
        if (type(config.device_ids) == int and config.device_ids == -1) or (
            type(config.device_ids) == list and len(config.device_ids) == 0
        ):
            config.device_ids = list(range(torch.cuda.device_count()))
        elif isinstance(config.device_ids, int):
            config.device_ids = [config.device_ids]
        elif isinstance(config.device_ids, str):
            config.device_ids = [int(m) for m in config.device_ids.split(",")]
        print(f"Using cuda devices: {config.device_ids}")

    return config
