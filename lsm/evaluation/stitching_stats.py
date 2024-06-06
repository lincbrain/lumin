import numpy as np
from typing import Optional

import pandas as pd
from scipy import stats

from lsm.evaluation.plotting import box_plot, line_plot


class StitchingAnalysis:
    def __init__(
        self,
        metric_dict: dict,
        metric_name: str,
        var_name: str,
        save_path: str,
        resolution: Optional[int] = 300,
    ):
        self.metric_dict = metric_dict
        self.metric_name = metric_name
        self.var_name = var_name
        self.save_path = save_path
        self.resolution = resolution

        # create dataframe
        self.dataframe = self._to_dataframe()

    def _to_dataframe(self):
        # convert metric dictionary to pandas dataframe
        # compute descriptive statistics
        df = (
            pd.DataFrame.from_dict(self.metric_dict, orient="index")
            .transpose()
            .melt(var_name=self.var_name, value_name=self.metric_name)
        )

        return df

    def descriptive_stats(self):
        # display descriptive statistics
        stats_summary = self.dataframe.groupby(self.var_name)[self.metric_name].agg(
            ["mean", "std", "var"]
        )
        print(f"Descriptive statistics: {stats_summary}\n")
        return stats_summary

    def oneway_anova(self):
        # perform one-way anova
        metric_list = [
            self.dataframe[self.dataframe[self.var_name] == chunk][self.metric_name]
            for chunk in self.dataframe[self.var_name].unique()
        ]
        anova_result = stats.f_oneway(*metric_list)
        print(f"ANOVA: {anova_result}\n")

        return anova_result

    def all_analysis(self):
        all_stats = self.descriptive_stats()
        anova_result = self.oneway_anova()

        # make plots
        box_plot(
            data=self.dataframe,
            metric_name=self.metric_name,
            var_name=self.var_name,
            save_path=self.save_path,
            resolution=self.resolution,
        )
        line_plot(
            data=all_stats,
            metric_name=self.metric_name,
            var_name=self.var_name,
            save_path=self.save_path,
            resolution=self.resolution,
        )

        return all_stats, anova_result


if __name__ == "__main__":
    iou_scores = {
        "20": np.random.rand(100),
        "30": np.random.rand(100),
        "40": np.random.rand(100),
        "50": np.random.rand(100),
        "60": np.random.rand(100),
    }

    analysis = StitchingAnalysis(
        metric_dict=iou_scores, metric_name="IoU", var_name="chunk size", save_path="./"
    )
    _ = analysis.all_analysis()
