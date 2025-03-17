# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

import pandas as pd
from IPython.display import HTML


def plot_memory_timeline(trace_file):
    """Plots html output of torch profiler memory trace
    For use within Jupyter Notebook only!
    See https://pytorch.org/docs/main/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline

    Args:
        trace_file: path to html export of torch profiler memory timeline
    """
    with open(trace_file) as f:
        return HTML(f.read())


# These are the (unlabeled) columns in the json export of a torch profiler memory timeline trace
COL_NAMES = [
    "Parameter",
    "Optimizer_State",
    "Input",
    "Temporary",
    "Activation",
    "Gradient",
    "Autograd_Detail",
    "Unknown",
]


def create_mem_df(mem_trace, units="GB"):
    """Create dataframe from json export of torch profiler CUDA memory timeline trace
    Columns per COL_NAMES, in units of MB
    These are the (unlabeled) columns in the json export of a torch profiler memory timeline trace but can be
    inferred (and confirmed) by comparing the plots of the json export with the plots of the html export

    E.g., df.plot(kind="area", stacked=True, ylabel="MB")

    See https://pytorch.org/docs/main/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline
    Args:
        mem_trace: path to json export of torch profiler memory timeline
        units: "MB" or "GB"
    """
    df = pd.read_json(mem_trace).T.explode(0)

    def _convert_to_units(df, col):
        return df[col] / 1024 ** (3 if units == "GB" else 2)

    convert_cols_to_MB = {col: partial(_convert_to_units, col=col) for col in COL_NAMES}

    df = pd.DataFrame(
        [l[1:] for l in df.iloc[:, 1].to_list()], columns=COL_NAMES
    ).assign(**convert_cols_to_MB)
    df["Total"] = df.sum(axis=1)
    return df


def show_memory_stats(df, stats=["mean", "min", "50%", "max"]):
    """Show summary statistics for torch profiler CUDA memory timeline trace
    Args:
        df: dataframe created by create_mem_df
        stats: list of statistics to show. Valid stats are "mean", "min", "25%", "50%", "75%", "max"

    """
    mem_sum = (
        df.describe()
        .loc[stats]
        .rename(index={"50%": "median"})
        .style.format(precision=1, thousands=",")
    )

    return mem_sum
