import os
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Plots:
    """
    class for all plots
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metric_name: str,
        var_name: str,
        save_path: str,
        resolution: Optional[int] = 300,
    ):
        self.data = data
        self.metric_name = metric_name
        self.var_name = var_name
        self.save_path = save_path
        self.resolution = resolution

    def box_plot(self):
        # create a box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.var_name, y=self.metric_name, data=self.data)
        plt.title(f"Distribution of {self.metric_name} Across Chunk Sizes")
        plt.ylabel(self.metric_name)
        plt.xlabel(self.var_name)

        # save plots
        fpath = os.path.join(
            self.save_path, f"{self.metric_name}_{self.var_name}_boxplot.png"
        )
        plt.savefig(fpath, dpi=self.resolution)
        plt.close()

    def line_plot(self):
        # line plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.data, markers=True, dashes=False)
        plt.fill_between(
            self.data.index,
            self.data["mean"] - self.data["std"],
            self.data["mean"] + self.data["std"],
            alpha=0.2,
        )
        plt.title(
            f"Mean {self.metric_name} with Standard Deviation Across {self.var_name}"
        )
        plt.xlabel(self.var_name)
        plt.ylabel(self.metric_name)

        # save plots
        fpath = os.path.join(
            self.save_path, f"{self.metric_name}_{self.var_name}_lineplot.png"
        )
        plt.savefig(fpath, dpi=self.resolution)
        plt.close()

    def bar_plot(self):
        # bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=self.var_name, y=self.metric_name, data=self.data, palette="Blues_d"
        )
        plt.title(f"Distribution of {self.metric_name} across {self.var_name}")
        plt.xlabel(self.var_name)
        plt.ylabel(self.metric_name)

        # save plots
        fpath = os.path.join(
            self.save_path, f"{self.metric_name}_{self.var_name}_barplot.png"
        )
        plt.savefig(fpath, dpi=self.resolution)
        plt.close()

    def point_plot(self):
        # point plot
        plt.figure(figsize=(10, 6))
        sns.pointplot(x=self.var_name, y=self.metric_name, data=self.data, ci=None)
        plt.title(f"Point Plot of {self.metric_name} across {self.var_name}")
        plt.xlabel(self.var_name)
        plt.ylabel(self.metric_name)

        # save plots
        fpath = os.path.join(
            self.save_path, f"{self.metric_name}_{self.var_name}_pointplot.png"
        )
        plt.savefig(fpath, dpi=self.resolution)
        plt.close()
