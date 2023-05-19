# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import csv
import os.path
import warnings
from math import inf
from typing import List

import numpy as np
import torch

from benchmarks import Benchmark, read_benchmark
from utils import Stats

HEADER = (
    "Benchmark",
    "Width",
    "Seed",
    "Result",
    "Partitions",
    "Prune",
    "Error_check",
    "Transitions",
    "Transitions_Pruned",
    "Error",
    "Est_Mean_se",
    "Est_Max_se",
    "New_Mean_Error",
    "New_Max_Error",
    "New_Min_Error",
    "Error_std",
    "Template",
    "Tot_Time",
    "Learner_Time",
    "Certifier_Time",
    "Abstraction_Time",
    "Verification_Time",
)


class CSVWriter:
    """Class for writing results to csv file."""

    def __init__(self, filename: str, headers: List[str]) -> None:
        """Initializes CSVWriter.

        If the file does not exist, it will be created here
        and the header will be written to it.

        Args:
            filename (str): filename of csv file.
            headers (List[str]): headers for csv file
        """
        self.headers = headers
        self.filename = filename
        if not os.path.isfile(self.filename):
            self.write_header_to_csv()

    def write_header_to_csv(self):
        """Creates csv file and writes a header to it."""
        with open(self.filename, "a") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.headers, delimiter=",", lineterminator="\n"
            )
            writer.writeheader()

    def write_row_to_csv(self, values: List):
        """Writes values to row in CSV file."""
        if len(values) != len(self.headers):
            warnings.warn("More values to write than columns in csv.")
        with open(self.filename, "a") as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow(values)


class Analyser:
    """Object to calculate and record stats of abstraction."""

    def __init__(self, abstraction) -> None:
        """initializes Analyser.

        Args:
            abstraction (NeuralAbstraction): abstraction to analyse
        """
        self.abstraction = abstraction
        self.filename = None

    def record(
        self,
        filename: str,
        config,
        res: str,
        tot_time: float,
        times: tuple[float, float, float, float],
        errors,
        transitions,
        transitions_pruned,
        seed,
    ):
        """records results of abstraction to csv file.

        Args:
            filename (str): csv file to write to
            config (_type_): program configuration
            res (str): result of synthesis
            T (float): duration of synthesis
        """
        self.filename = filename + ".csv"
        headers = HEADER
        writer = CSVWriter(self.filename, headers)
        mse, Mse = None, None
        p = len(self.abstraction.locations)
        if not res:
            errors = Stats(inf, inf, inf, inf)
        writer.write_row_to_csv(
            [
                self.abstraction.benchmark.name,
                np.array(self.abstraction.nets[0].width),
                seed,
                res,
                p,
                config.prune,
                config.error_check,
                transitions,
                transitions_pruned,
                np.array(self.abstraction.error),
                None,
                None,
                errors.mean,
                errors.max,
                errors.min,
                errors.std,
                config.template,
                tot_time,
                times[0],
                times[1],
                times[2],
                times[3],
            ]
        )

    def record_failure(
        self,
        filename,
        config,
        res: str,
        tot_time: float,
        times: tuple[float, float, float, float],
        seed,
    ):
        self.filename = filename + ".csv"
        headers = HEADER
        writer = CSVWriter(self.filename, headers)
        p = None
        errors = Stats(inf, inf, inf, inf)
        writer.write_row_to_csv(
            [
                read_benchmark(config.benchmark).name,
                np.array(config.widths),
                seed,
                res,
                p,
                config.prune,
                config.error_check,
                None,
                None,
                None,
                None,
                None,
                errors.mean,
                errors.max,
                errors.min,
                errors.std,
                config.template,
                tot_time,
                times[0],
                times[1],
                times[2],
                times[3],
            ]
        )

    def estimate_errors(self) -> float:
        """Estimates mean and max error of abstraction.

        Returns:
           mse float: mean squared error
           Mse float: max squared error
        """
        return (
            self.error_calculator.estimate_mse(),
            self.error_calculator.estimate_max_square_error(),
        )


class ErrorCalculator:
    """Class for estimating errors of abstractions."""

    def __init__(self, benchmark: Benchmark, abstraction, p=10000) -> None:
        """initializes ErrorCalculator.

        Args:
            benchmark (Benchmark): _description_
            abstraction (_type_): _description_
            p (int, optional): _description_. Defaults to 10000.
        """
        self.benchmark = benchmark
        self.abstraction = abstraction
        self.p = p

    def estimate_mse(self) -> float:
        """Estimates mean square error of abstraction"""
        data = self.benchmark.domain.generate_data(self.p)
        y = self.benchmark.f(data)
        y_est = self.abstraction(data).T
        mse_error = torch.nn.MSELoss(reduction="mean")
        mse = mse_error(y, y_est)
        return mse.item()

    def estimate_max_square_error(self) -> float:
        """Estimates max square error of abstraction."""
        data = self.benchmark.domain.generate_data(self.p)
        y = self.benchmark.f(data)
        y_est = self.abstraction(data).T
        mse_error = torch.nn.MSELoss(reduction="none")
        max_se = mse_error(y, y_est).max()
        return max_se.item()
