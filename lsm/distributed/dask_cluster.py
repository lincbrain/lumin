import os
import yaml
import pathlib

import dask
import distributed
import dask_jobqueue
from distributed import Client, LocalCluster


def _config_path(config_name):
    """
    constructs the path to a dask configuration file located in the user's home directory

    Args:
        config_name (str): name of the configuration file

    Returns:
        (str): The full path to the Dask configuration file
    """
    return str(pathlib.Path.home()) + "/.config/dask/" + config_name


def _modify_dask_config(
    cfg,
    config_name=f"distributed_dask_config.yaml",
):
    """
    modifies the dask configuration and saves the updated configuration to a specified file

    Args:
        cfg (dict): the configuration dictionary to be set for dask
        config_name (str, optional): the name of the configuration file to save the updated configuration. Defaults to "distributed_dask_config.yaml"
    """
    dask.config.set(cfg)
    with open(_config_path(config_name), "w") as f:
        yaml.dump(dask.config.config, f, default_flow_style=False)


def _remove_config_file(
    config_name="distributed_dask_config.yaml",
):
    """
    removes the specified Dask configuration file if it exists

    Args:
        config_name (str, optional): the name of the configuration file to be removed. Defaults to "distributed_dask_config.yaml"
    """
    config_path = _config_path(config_name)
    if os.path.exists(config_path):
        os.remove(config_path)


class CustomCluster(dask_jobqueue.LSFCluster):
    """
    customize the LSFCluster from dask_jobqueue for use with dask,
    configuring it with specific CPU and worker parameters, and provide
    additional functionality for cluster management
    """

    def __init__(
        self,
        ncpus,
        min_workers,
        max_workers,
        config={},
        config_name="distributed_cellpose_dask_config.yaml",
        persist_config=False,
        **kwargs,
    ):
        """
        Args:
            ncpus (int): number of CPUs per job
            min_workers (int): minimum number of workers for adaptive scaling
            max_workers (int): maximum number of workers for adaptive scaling
            config (dict, optional): configuration dictionary for dask. Defaults to an empty dictionary
            config_name (str, optional): name of the configuration file to be saved. Defaults to "distributed_cellpose_dask_config.yaml"
            persist_config (bool, optional): flag indicating whether to persist the dask configuration file. Defaults to False
        """
        self.config_name = config_name
        self.persist_config = persist_config
        scratch_dir = f"{os.getcwd()}/"
        scratch_dir += f".{os.environ['USER']}_distributed_cellpose/"
        config_defaults = {
            "temporary-directory": scratch_dir,
            "distributed.comm.timeouts.connect": "180s",
            "distributed.comm.timeouts.tcp": "360s",
        }
        config = {**config_defaults, **config}
        _modify_dask_config(cfg=config, config_name=config_name)

        job_script_prologue = [
            f"export MKL_NUM_THREADS={2*ncpus}",
            f"export NUM_MKL_THREADS={2*ncpus}",
            f"export OPENBLAS_NUM_THREADS={2*ncpus}",
            f"export OPENMP_NUM_THREADS={2*ncpus}",
            f"export OMP_NUM_THREADS={2*ncpus}",
        ]

        # init local and log directories
        if "local_directory" not in kwargs:
            kwargs["local_directory"] = scratch_dir
        if "log_directory" not in kwargs:
            log_dir = f"{os.getcwd()}/dask_worker_logs_{os.getpid()}/"
            pathlib.Path(log_dir).mkdir(parents=False, exist_ok=True)
            kwargs["log_directory"] = log_dir

        class quietLSFJob(dask_jobqueue.lsf.LSFJob):
            cancel_command = "bkill -d"

        super().__init__(
            ncpus=ncpus,
            processes=1,
            cores=1,
            memory=str(15 * ncpus) + "GB",
            mem=int(15e9 * ncpus),
            job_script_prologue=job_script_prologue,
            job_cls=quietLSFJob,
            **kwargs,
        )
        self.client = distributed.Client(self)
        print("Cluster dashboard link: ", self.dashboard_link)

        # set adaptive cluster bounds for managing worker processes
        self.adapt_cluster(min_workers, max_workers)

    def __enter__(self):
        """
        supports the context management protocol to ensure proper resource handling
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        supports the context management protocol to ensure proper
        cleanup of resources and removal of the configuration file if
        persist_config is False.

        Args:
            exc_type: exception type
            exc_value: exception value
            traceback: traceback object
        """
        if not self.persist_config:
            _remove_config_file(self.config_name)
        self.client.close()
        super().__exit__(exc_type, exc_value, traceback)

    def adapt_cluster(self, min_workers, max_workers):
        """
        sets the adaptive cluster bounds for managing worker processes

        Args:
            min_workers (int): minimum number of workers for adaptive scaling
            max_workers (int): maximum number of workers for adaptive scaling
        """
        _ = self.adapt(
            minimum_jobs=min_workers,
            maximum_jobs=max_workers,
            interval="10s",
            wait_count=6,
        )

    def change_worker_attributes(
        self,
        min_workers,
        max_workers,
        **kwargs,
    ):
        """
        changes the attributes of the workers and adapts the cluster accordingly

        Args:
        - min_workers (int): minimum number of workers for adaptive scaling
        - max_workers (int): maximum number of workers for adaptive scaling
        """
        self.scale(0)
        for k, v in kwargs.items():
            self.new_spec["options"][k] = v
        self.adapt_cluster(min_workers, max_workers)


def cluster(f):
    """
    decorator to execute a function with a dask cluster created using a `CustomCluster`

    this decorator ensures that a `CustomCluster` is initialized and passed
    to the decorated function  as a `cluster` keyword argument, unless a cluster
    is already provided in the `kwargs`

    the cluster is managed using a context manager to ensure proper resource handling

    Args:
        f (function): the function to be decorated

    Returns:
        function: a wrapped version of the input function that will execute within the context of a dask cluster
    """

    @functools.wraps(f)
    def create_cluster(*args, **kwargs):
        if not "cluster" in kwargs:
            F = lambda x: x in kwargs["cluster_kwargs"]
            cluster_constructor = CustomCluster
        with cluster_constructor(**kwargs["cluster_kwargs"]) as cluster:
            kwargs["cluster"] = cluster
            return func(*args, **kwargs)

        return f(*args, **kwargs)

    return create_cluster


if __name__ == "__main__":
    # test cluster class
    # we just need to specify number of cpus, min / max workers
    cluster_kwargs = {
        "ncpus": 2,
        "min_workers": 10,
        "max_workers": 100,
    }
    with CustomCluster(**cluster_kwargs) as cluster:
        print(f"initializing cluster...")
        cluster = CustomCluster(**cluster_kwargs)
        print(f"cluster: {cluster}")
        cluster.close()
