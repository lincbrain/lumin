import numpy as np
from typing import Optional

import pandas as pd
from scipy import stats

from lsm.evaluation.plotting import Plots


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
        if self.metric_name not in self.dataframe.columns:
            raise ValueError(f"{self.metric_name} not computed.")

        data = self.dataframe[self.metric_name].values

        mean = np.mean(data)
        # bessel's correction (ddof=1)
        std = np.std(data, ddof=1)
        var = np.var(data, ddof=1)

        stats_summary = pd.DataFrame(
            {"mean": [mean], "std": [std], "var": [var]}, index=[self.metric_name]
        )

        return stats_summary

    def kruskal_test(self):
        # perform kruskal h-test
        # since our sample size is just 1
        if self.var_name not in self.dataframe.columns:
            raise ValueError(
                f"Column '{self.var_name}' does not exist in the dataframe."
            )
        if self.metric_name not in self.dataframe.columns:
            raise ValueError(
                f"Column '{self.metric_name}' does not exist in the dataframe."
            )

        grouped_data = self.dataframe.groupby(self.var_name)[self.metric_name].apply(
            list
        )

        f_val, p_val = stats.kruskal(*grouped_data)

        if p_val < 0.05:
            print(
                f"null hypothesis rejected, ie: population medians are unequal (p < 0.05)\n"
            )

        return p_val

    def all_analysis(self):
        all_stats = self.descriptive_stats()
        kruskal_result = self.kruskal_test()

        # make plots
        plot_obj = Plots(
            data=self.dataframe,
            metric_name=self.metric_name,
            var_name=self.var_name,
            save_path=self.save_path,
            resolution=self.resolution,
        )
        plot_obj.bar_plot()
        plot_obj.point_plot()

        return all_stats, kruskal_result


if __name__ == "__main__":
    iou_scores = {
        "20": np.random.rand(1),
        "30": np.random.rand(1),
        "40": np.random.rand(1),
        "50": np.random.rand(1),
        "60": np.random.rand(1),
    }

    analysis = StitchingAnalysis(
        metric_dict=iou_scores, metric_name="IoU", var_name="chunk size", save_path="./"
    )
    _ = analysis.all_analysis()
