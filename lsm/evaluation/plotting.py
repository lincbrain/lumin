import os
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def box_plot(
    data: pd.DataFrame,
    metric_name: str,
    var_name: str,
    save_path: str,
    resolution: Optional[int] = 300,
):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=var_name, y=metric_name, data=data)
    plt.title(f"Distribution of {metric_name} Across Chunk Sizes")
    plt.xlabel(var_name)
    plt.ylabel(metric_name)

    # save plots
    save_path = os.path.join(save_path, f"{metric_name}_{var_name}_boxplot.png")
    plt.savefig(save_path, dpi=resolution)
    plt.close()


def line_plot(
    data: pd.DataFrame,
    metric_name: str,
    var_name: str,
    save_path: str,
    resolution: Optional[int] = 300,
):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, markers=True, dashes=False)
    plt.fill_between(
        data.index,
        data["mean"] - data["std"],
        data["mean"] + data["std"],
        alpha=0.2,
    )
    plt.title(f"Mean {metric_name} with Standard Deviation Across {var_name}")
    plt.xlabel(var_name)
    plt.ylabel(metric_name)

    # save plots
    save_path = os.path.join(save_path, f"{metric_name}_{var_name}_lineplot.png")
    plt.savefig(save_path, dpi=resolution)
    plt.close()
