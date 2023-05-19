from math import floor, log10

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def sig_fig(x, n=3):
    try:
        s = round(x, n - int(floor(log10(abs(x)))))
    except ValueError:
        s = x
    return s


# Read the data
def table_prune():
    df = pd.read_csv("results/results.csv")
    df = df[df["Template"] != "sig"]
    df = df.filter(
        [
            "Benchmark",
            "Template",
            "Seed",
            "Prune",
            "Partitions",
            "Transitions",
            "Transitions_Pruned",
            "Error",
            "Abstraction_Time",
            "Verification_Time",
        ]
    )
    df_prune = df[df["Prune"] == True].reset_index()
    df_no_prune = df[df["Prune"] == False].reset_index()
    df_prune["Delta_Abstraction_Time"] = (
        df_prune["Abstraction_Time"] - df_no_prune["Abstraction_Time"]
    )
    df_prune["Delta_Verification_Time"] = (
        df_prune["Verification_Time"] - df_no_prune["Verification_Time"]
    )

    grouped = df_prune.groupby(["Benchmark", "Template"])
    table = grouped.aggregate(
        {
            "Delta_Abstraction_Time": ["mean", "max", "min"],
            "Delta_Verification_Time": ["mean", "max", "min"],
        }
    )
    table.to_latex(
        "results/pruning.tex",
        float_format="%.2f",
        bold_rows=True,
    )


def table_main():
    df = pd.read_csv("results/results.csv")

    df["Total_Time"] = (
        df["Learner_Time"]
        + df["Certifier_Time"]
        + df["Verification_Time"]
        + df["Abstraction_Time"]
    )

    df = df.drop(df[df["Result"] != "S"].index)
    df["Error"] = df["Error"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
    df["Error_1_Norm"] = df["Error"].apply(lambda x: np.linalg.norm(x, ord=1))

    df = df.drop(df[df["Prune"] == True].index)

    grouped = df.groupby(["Benchmark", "Template", "Width"])

    table = grouped.aggregate(
        {
            "Error_1_Norm": ["min", "mean", "max"],
            "Partitions": ["min", "mean", "max"],
            # "Transitions": ["min","mean",  "max"],
            "Total_Time": ["min", "mean", "max"],
        }
    )
    # table = table.round(2)
    # table = table.applymap(lambda x: sig_fig(x))
    table = table.loc[
        [
            "Water-tank",
            "Non-Lipschitz1",
            "Non-Lipschitz2",
            "Water-tank-4d",
            "Water-tank-6d",
            "NODE1",
        ],
        ["pwc", "pwa", "sig"],
        :,
        :,
    ]
    table.rename(
        {
            "Error_1_Norm": "$||\epsilon||_1$",
            "Partitions": "$M$",
            # "Transitions": "Transitions",
            "Total_Time": "$T$",
            "Width": "$N$",
            "mean": "$\mu$",
            "min": "$\min$",
            "max": "$\max$",
        },
        axis=1,
        inplace=True,
    )
    table.to_latex(
        "results/main_tab.tex",
        float_format="%.3g",
        bold_rows=True,
        escape=False,
        multicolumn_format="c",
    )


def table_timings():
    df = pd.read_csv("results/results.csv")

    df = df.drop(df[df["Prune"] == True].index)
    df = df.drop(df[df["Result"] != "S"].index)

    grouped = df.groupby(["Benchmark", "Template"])

    table = grouped.aggregate(
        {
            "Learner_Time": ["min", "mean", "max"],
            "Certifier_Time": ["min", "mean", "max"],
            # "Abstraction_Time": ["min", "mean", "max"],
            "Verification_Time": ["min", "mean", "max"],
        }
    )
    # table = table.applymap(lambda x: sig_fig(x))
    table = table.loc[
        [
            "Water-tank",
            "Non-Lipschitz1",
            "Non-Lipschitz2",
            "Water-tank-4d",
            "Water-tank-6d",
            "NODE1",
        ],
        ["pwc", "pwa", "sig"],
        :,
    ]
    table.rename(
        {
            "Learner_Time": "$T_L$",
            "Certifier_Time": "$T_C$",
            "Abstraction_Time": "$T_A$",
            "Verification_Time": "$T_f$",
            "mean": "$\mu$",
            "min": "$\min$",
            "max": "$\max$",
        },
        axis=1,
        inplace=True,
    )
    table.to_latex(
        "results/time_tab.tex",
        float_format="%.3g",
        bold_rows=True,
        escape=False,
        multicolumn_format="c",
    )


def table_error_check():
    results = pd.read_csv("results/error-refine.csv")
    results["New_Mean_Error"] = results["New_Mean_Error"].apply(
        lambda x: np.fromstring(x[1:-1], sep=" ")
    )
    results["Error_1_Norm"] = results["New_Mean_Error"].apply(
        lambda x: np.linalg.norm(x, ord=1)
    )
    results["Error_1_Norm"] = results["Error_1_Norm"].mask(results["Result"] == "False")
    results.rename({"Error_check": "Error Refinement"}, inplace=True, axis=1)
    results.rename({"Error_check": "Error Refinement"}, inplace=True, axis=1)
    table = results.groupby(["Template", "Error Refinement"])

    table = table.aggregate(
        {
            "Certifier_Time": ["mean"],
            "Verification_Time": ["mean"],
            "Error_1_Norm": ["mean"],
        }
    )
    table = table.applymap(lambda x: sig_fig(x))
    table["Success Rate"] = results.groupby(["Template", "Error Refinement"]).apply(
        lambda x: x["Result"].str.contains("S").sum() / len(x["Result"])
    )
    table.rename(
        {
            "Certifier_Time": "$\\bar{T}_C$",
            "Verification_Time": "$\\bar{T}_f$",
            "mean": "$\mu$",
            "Success Rate": "Success Rate",
            "Error_1_Norm": "$ ||\\bar{\epsilon}||_1$",
        },
        axis=1,
        inplace=True,
    )
    # table = table.transpose()
    table.to_latex(
        "results/error_check.tex",
        float_format="%.3g",
        bold_rows=True,
        escape=False,
        multicolumn_format="c",
    )


def report_failures():
    df = pd.read_csv("results/results.csv")
    failures = df[df["Result"] != "S"]
    total_failures = len(failures)

    # If an overall timeout occurs, there is no seed
    # If all experiments fail in flowpipe propagation, there is a seed as we return the last experiment to fail
    overall_timeouts = failures[failures["Seed"].isnull()]
    other_timeouts = failures[failures["Seed"].notnull()]
    print("Total Failures: {}".format(total_failures))
    print("\n Flowpipe Propagation Failures: {}".format(len(other_timeouts)))
    for i, row in other_timeouts.iterrows():
        print("Benchmark {}, Template: {}".format(row["Benchmark"], row["Template"]))

    print("\n Overall Timeout Failures: {}".format(len(overall_timeouts)))
    for i, row in overall_timeouts.iterrows():
        print("Benchmark {}, Template: {}".format(row["Benchmark"], row["Template"]))


def total_computation_time():
    pass


if __name__ == "__main__":
    table_main()
    table_timings()
    table_error_check()
    report_failures()
