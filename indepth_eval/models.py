from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from pydantic import BaseModel
import seaborn as sns
import matplotlib.pyplot as plt


ESTIM_LABEL_MAP = {
    "AvelinoEstimator": "AVE",
    "HaratianEstimator": "H-DC",
    "EDoKEstimator": "EDoK",
    "SocialGraphEstimator": "SNet"
}


class ContribClassificationAnalysis(BaseModel):
    true_positive_contribs: List[str] = []  # Those who were correctly INCLUDED in the TF count
    false_positive_contribs: List[str] = []  # Those who were wrongly INCLUDED in the TF count
    false_negative_contribs: List[str] = []  # Those who were wrongly EXCLUDED from the TF count

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class EstimatorPerf(BaseModel):
    # Map of project_name to Tuple[Norm_Abs_Error, Contrib_List_Comparison_Breakdown]
    estimate_breakdowns: Dict[str, Tuple[float, ContribClassificationAnalysis]] = dict()
    norm_mean_abs_err: float = 0.0  # should only be set through parent container calc_metrics method


class EstimatorPerfContainer(BaseModel):
    project_count: int
    # Estimator Key -> Estimator performance
    performance_entries: Dict[str, EstimatorPerf] = dict()

    def calc_metrics(self):
        for entry in self.performance_entries.values():
            # Calculate NMAE score
            total_norm_abs_err = sum([i[0] for i in entry.estimate_breakdowns.values()])
            entry.norm_mean_abs_err = total_norm_abs_err / len(entry.estimate_breakdowns.keys())

            # Calculate Precision, Recall F-measure
            classification_analysis: ContribClassificationAnalysis
            for _, classification_analysis in entry.estimate_breakdowns.values():
                tp = len(classification_analysis.true_positive_contribs)
                fp = len(classification_analysis.false_positive_contribs)
                fn = len(classification_analysis.false_negative_contribs)
                # protect against div by 0
                if tp > 0 or (fp > 0 and fn > 0):
                    precision = tp/(tp+fp)
                    recall = tp/(tp+fn)
                    classification_analysis.precision = precision
                    classification_analysis.recall = recall
                    if precision > 0 and recall > 0:
                        classification_analysis.f1 = (2*precision*recall)/(precision+recall)
                    else:
                        classification_analysis.f1 = 0.0

    def plot_classification_metrics(self, output_path: Path):
        collated_precision = []
        collated_recall = []
        collated_f1 = []
        for estimator_key, perf_data in self.performance_entries.items():
            collated_precision += [
                {"Estimator": ESTIM_LABEL_MAP[estimator_key.split("-")[0]], "Precision": perf_entry.precision}
                for _, perf_entry in perf_data.estimate_breakdowns.values()
            ]
            collated_recall += [
                {"Estimator": ESTIM_LABEL_MAP[estimator_key.split("-")[0]], "Recall": perf_entry.recall}
                for _, perf_entry in perf_data.estimate_breakdowns.values()
            ]
            collated_f1 += [
                {"Estimator": ESTIM_LABEL_MAP[estimator_key.split("-")[0]], "F1": perf_entry.f1}
                for _, perf_entry in perf_data.estimate_breakdowns.values()
            ]

        precision_df = pd.DataFrame(collated_precision)
        recall_df = pd.DataFrame(collated_recall)
        f1_df = pd.DataFrame(collated_f1)

        precision_plot = sns.boxplot(
            data=precision_df,
            x="Estimator",
            y="Precision",
            width=0.4,
            linewidth=1.25,
            palette=sns.color_palette("pastel", 1)
        )
        precision_plot.set_xticklabels(precision_plot.get_xticklabels())
        plt.xlabel("Estimator", fontsize=16)
        plt.ylabel("Precision", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(output_path.joinpath("precision-comparison.pdf"), bbox_inches='tight')
        plt.close("all")

        recall_plot = sns.boxplot(
            data=recall_df,
            x="Estimator",
            y="Recall",
            width=0.4,
            linewidth=1.25,
            palette=sns.color_palette("pastel", 1)
        )
        recall_plot.set_xticklabels(recall_plot.get_xticklabels())
        plt.xlabel("Estimator", fontsize=16)
        plt.ylabel("Recall", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(output_path.joinpath("recall-comparison.pdf"), bbox_inches='tight')
        plt.close("all")

        f1_plot = sns.boxplot(
            data=f1_df,
            x="Estimator",
            y="F1",
            width=0.4,
            linewidth=1.25,
            palette=sns.color_palette("pastel", 1)
        )
        f1_plot.set_xticklabels(f1_plot.get_xticklabels())
        plt.xlabel("Estimator", fontsize=16)
        plt.ylabel("F1", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(output_path.joinpath("f1-comparison.pdf"), bbox_inches='tight')
        plt.close("all")

    def get(self, e_key: str) -> Optional[EstimatorPerf]:
        return self.performance_entries.get(e_key, None)
