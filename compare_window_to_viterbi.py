import json
import typing as tp
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import colormaps
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison


class PlotType:
    def __init__(
        self,
        name: str,
        folder: Path,
        model_order: tp.Dict[str, int],
        metrics_labels: tp.List[str] = [
            "Unreduced Kappa",
            "Reduced Kappa",
            "Unreduced F1",
            "Reduced F1",
        ],
        dataset_order: tp.Dict[str, int] = {
            "three easy nodrift": 1,
            "three easy drift": 2,
            "three hard nodrift": 3,
            "three hard drift": 4,
            "five easy nodrift": 5,
            "five easy drift": 6,
            "five hard nodrift": 7,
            "five hard drift": 8,
        },
    ):
        self.name = name
        self.folder = folder
        self.model_order = model_order
        self.metrics_labels = metrics_labels
        self.dataset_order = dataset_order

        self.naive_folder = Path("../3.1_results/output/")
        self.viterbi_folder = Path("../3.2/output")
        self.cmap = [colormaps["tab10"](a) for a in np.linspace(0, 1, 10)]
        self.bracket_h = 0.02

        self.metric_path_list: tp.Optional[tp.List[Path]] = None
        self.metrics_df: tp.Optional[pd.DataFrame] = None
        self.statistics: tp.Optional[tp.Dict[str, pd.DataFrame]] = None

    def run(self):
        # Get list of all metric filepaths
        self.get_total_file_list()

        # Parse all data into a dataframe
        df = self.parse_all_data()

        for dataset in df["Dataset"].unique():
            # Subset the dataframe to the current dataset
            subset = df[df["Dataset"] == dataset]

            # Get statistics for the dataset
            self.get_statistics(subset, dataset)

            for metric in self.metrics_labels:
                # Make the plot
                self.make_plot(subset, dataset, metric)

    def get_total_file_list(self) -> tp.List[Path]:
        """Get all the metric filepaths for the current plot type.

        Returns:
            tp.List[Path]: List of all metric filepaths.
        """

        metric_path_list = self.get_file_list(self.folder)

        # Add the metrics of the previous experiment
        if self.name == "Viterbi":
            metric_path_list += self.get_file_list(
                self.naive_folder, filter_to_top=True
            )
        elif self.name == "Window":
            metric_path_list += self.get_file_list(
                self.viterbi_folder, filter_to_top=True
            )

        self.metric_path_list = metric_path_list

        return metric_path_list

    def parse_all_data(self):
        # Check if we have a list of metric paths
        if self.metric_path_list is None:
            raise ValueError("No metric path list!")

        all_data = []

        # Parse all the data
        for metric_path in self.metric_path_list:
            # Get a pretty model name
            model_name = metric_path.parent.parent.name.replace("_", " ")

            # Clarify the model name, adding Viterbi if it's from the viterbi experiement
            if "3.2" in str(metric_path):
                model_name = "Viterbi " + model_name

            if "3.3" in str(metric_path):
                model_name = "Viterbi " + model_name

            # Get a pretty dataset name
            dataset = metric_path.parent.parent.parent.name.replace("_", " ")

            # Open and read the metrics and add to the data list
            with open(metric_path, "r") as f:
                data = json.loads(f.read())
                data = [data[label] for label in self.metrics_labels]

            all_data.append([model_name, dataset, *data])

        # Convert to a dataframe
        df = pd.DataFrame(all_data, columns=["Model", "Dataset", *self.metrics_labels])

        self.metrics_df = df

        return df

    def make_plot(self, subset: pd.DataFrame, dataset: str, metric: str):
        num_models = len(subset["Model"].unique())
        bar_width = 0.8 / num_models

        if num_models % 2:
            raise ValueError("Odd number of models, something's wrong!")

        _, ax = plt.subplots(figsize=(8, 6))

        grouped_subset = (
            subset[["Model", metric]]
            .groupby("Model")
            .agg(["mean", "sem"])
            .reset_index()
            .sort_values(by="Model", key=lambda x: x.map(self.model_order))
        )
        substats = self.statistics[metric]

        ax.bar(
            x=grouped_subset["Model"],
            height=grouped_subset[metric]["mean"],
            yerr=grouped_subset[metric]["sem"],
            color=self.cmap[:num_models],
        )

        max_y = grouped_subset[metric]["mean"].max()
        lines_to_add = self.get_lines_to_add(substats)

        for line in lines_to_add:
            p_val = line[1]

            if p_val > 0.05:
                self.add_line_to_ax(
                    ax, line[0][0], line[0][1], max_y, self.bracket_h, "red", "NS"
                )
            elif p_val > 0.01:
                self.add_line_to_ax(
                    ax, line[0][0], line[0][1], max_y, self.bracket_h, "black", "*"
                )
            elif p_val > 0.001:
                self.add_line_to_ax(
                    ax, line[0][0], line[0][1], max_y, self.bracket_h, "black", "**"
                )

            max_y += 3 * self.bracket_h

        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(f"{metric} for each model")

        plt.savefig(self.get_figure_name(dataset, metric))
        plt.close()

    def add_line_to_ax(
        self,
        ax: plt.Axes,
        x1: float,
        x2: float,
        y: float,
        h: float,
        col: str,
        label: str,
    ):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], c=col)
        ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", color=col)

    def get_lines_to_add(self, substats: pd.DataFrame):
        index_dict = {model: idx for idx, model in enumerate(self.model_order)}

        lines_to_add = []

        for row, _ in substats.iterrows():
            row = [a.strip() for a in row]

            # Skip first row
            if "p-adj" in row:
                continue

            p_val = float(row[3])

            # If p_val is very significant, skip
            if p_val < 0.001:
                continue

            models = sorted([index_dict[model] for model in row[:2]])

            lines_to_add.append([models, p_val])

        return lines_to_add

    def get_statistics(
        self,
        subset: pd.DataFrame,
        dataset_name: str,
        write_to_file: bool = True,
    ):
        report = ""

        # statsmodels has issues with spaces in column names
        corrected_labels = [a.replace(" ", "_") for a in self.metrics_labels]

        subset = subset.rename(
            {a: b for a, b in zip(self.metrics_labels, corrected_labels)}, axis=1
        )

        stats_dict = {}

        # For each metric, get some statistics
        for metric in corrected_labels:
            model = ols(f"{metric} ~ C(Model)", data=subset).fit()

            mc = MultiComparison(subset[metric], subset["Model"]).tukeyhsd().summary()

            # Hack to get the data into a dataframe
            data_hack = StringIO(mc.as_csv())
            df = pd.read_csv(data_hack, sep=",", header=0)
            stats_dict[metric.replace("_", " ")] = df

            report += f"STATISTICS FOR {metric}\n"
            report += sm.stats.anova_lm(model, typ=1).to_string()
            report += "\n"
            report += f"Pairwise comparison for {dataset_name}"
            report += mc.as_text()
            report += "\n"

        if write_to_file:
            with open(self.get_report_name(dataset_name), "w") as f:
                f.write(report)

        self.statistics = stats_dict
        return stats_dict

    def get_bar_offset(idx: int, num_models: int, bar_width: float):
        return bar_width * (idx - ((num_models - 1) / 2))

    def get_figure_name(self, dataset: str, metric: str):
        return f"{self.name.lower().capitalize()}_{dataset}_{metric.replace(' ', '_')}_result_plot.svg"

    def get_report_name(self, dataset_name: str = None):
        return f"{self.name.lower().capitalize()}_{dataset_name}_result_statistics.txt"

    @staticmethod
    def get_file_list(folder: Path, filter_to_top: bool = False):
        file_list = list(folder.glob("**/metrics/*.json"))

        if not filter_to_top:
            return file_list
        else:
            return filter(
                lambda x: not any(
                    y in str(x) for y in ["LSTM", "DeepChannel", "ResNet"]
                ),
                file_list,
            )


BASEPLOT = PlotType(
    name="Naive",
    folder=Path("../3.1_results/output/"),
    model_order={
        "LSTM": 1,
        "DeepChannel": 2,
        "Simple CNN": 3,
        "Split CNN": 4,
        "ResNet": 5,
        "UNet": 6,
    },
)


VITERBI = PlotType(
    name="Viterbi",
    folder=Path("../3.2/output"),
    model_order={
        "Simple CNN": 1,
        "Split CNN": 2,
        "UNet": 3,
        "Viterbi Simple CNN": 4,
        "Viterbi Split CNN": 5,
        "Viterbi UNet": 6,
    },
)

WINDOW = PlotType(
    name="Window",
    folder=Path("../3.3/output"),
    model_order={
        "Viterbi Simple CNN": 1,
        "Viterbi Split CNN": 2,
        "Viterbi UNet": 3,
        "Viterbi Simple CNN Window": 4,
        "Viterbi Split CNN Window": 5,
        "Viterbi UNet Window": 6,
    },
)


if __name__ == "__main__":
    for plot_type in [BASEPLOT, VITERBI, WINDOW]:
        plot_type.run()
