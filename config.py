# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from dataclasses import dataclass
from enum import Enum
from typing import Union

import torch.optim as optim


class VerifierType(Enum):
    """Enum to store the type of the verifier"""

    Z3 = 1
    DREAL = 2

    @staticmethod
    def from_string(s: str) -> "VerifierType":
        """Converts a string to a VerifierType"""
        if s == "z3":
            return VerifierType.Z3
        elif s == "dreal":
            return VerifierType.DREAL
        else:
            raise ValueError("Invalid verifier type")


@dataclass
class Config:
    """Class to store program configuration parameters"""

    def __init__(self, args):
        self.benchmark = args.benchmark
        self.widths = args.width
        self.n_data = args.Ndata
        self.verifier = VerifierType.from_string(args.verifier)
        self.verbose: bool = not args.quiet
        # self.scalar: bool = args.scalar
        self.learning_mode = args.stopping_criterion["mode"]
        self.target_error = args.stopping_criterion["target-error"]
        self.loss_stop = float(args.stopping_criterion["loss-stop"])
        self.loss_grad_stop = float(args.stopping_criterion["loss-grad-stop"])
        self.loss_grad_grad_stop = float(args.stopping_criterion["loss-grad-grad-stop"])
        if args.optimizer["type"] == "AdamW":
            self.optimizer = optim.AdamW
            self.lr = float(args.optimizer["lr"])
        elif args.optimizer["type"] == "SGD":
            self.optimizer = optim.SGD
            self.lr = float(args.optimizer["lr"])
        self.timeout = args.timeout
        self.timeout_duration = int(args.timeout_duration)
        self.seed = args.seed
        self.repeat = args.repeat
        self.save_net = args.save_net
        self.output_type = args.output_type
        self.output_file = args.output_file
        self.results_file = args.results
        self.bounded_time = args.bounded_time
        self.time_horizon = args.time_horizon
        self.template = args.template
        self.initial = args.initial
        self.forbidden = args.forbidden
        self.spaceex = args.spaceex
        self.flowstar = args.flowstar
        self.prune = args.prune
        self.error_check = args.error_check
        self.n_procs = args.n_procs
        self.flowstar_config = FlowstarConfig(**args.flowstar_config)


@dataclass
class FlowstarConfig:
    """Configuration for Flow*"""

    time: float = 1.0
    step_mode: str = "adaptive"  # adaptive, fixed
    step_size: Union[tuple[float, float], float] = (
        0.001,
        0.1,
    )  # (min, max), if adaptive, else fixed value
    remainder_estimation: float = 0.001
    qr_precondition: bool = True
    order_mode: str = "fixed"  # fixed, adaptive
    order: Union[tuple[int, int], int] = 6  # if fixed, else (min, max)
