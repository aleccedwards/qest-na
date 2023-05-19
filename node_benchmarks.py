# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""This module contains scripts to create neural ode benchmarks"""

import torch

import benchmarks as bm
import cegis.nn as nn
import cli
import config
import neural_abstraction as na


class Synth:
    def __init__(self, config):
        self.config = config
        self.benchmark = bm.read_benchmark(self.config.benchmark)
        self.sigma = self.config.template

    def create_from_benchmark(self):
        S = self.benchmark.get_data(n=self.config.n_data)
        Sdot = self.benchmark.f(S)
        error = self.config.target_error
        if self.sigma == "sig":
            learner = nn.SigmoidNet(
                self.benchmark, self.config.widths, error, self.config
            )
        elif self.sigma == "tanh":
            learner = nn.TanhNet(self.benchmark, self.config.widths, error, self.config)
        else:
            raise NotImplementedError(
                "{} activation functions not implemented".format(self.sigma)
            )
        optim = torch.optim.AdamW(learner.model.parameters(), lr=1e-3)
        learner.learn(S, Sdot, optim, scheduler=None)
        return learner

    @staticmethod
    def learner_save(learner, filename):
        torch.save(learner.save(), filename)

    def learner_flowstar(self, learner, filename):
        out_file = filename
        if self.config.template == "sig":
            abstraction = na.SigmoidNeuralAbstraction(
                [learner], [0 for _ in range(self.benchmark.dimension)], self.benchmark
            )
        elif self.config.template == "tanh":
            abstraction = na.TanhNeuralAbstraction(
                [learner], [0 for _ in range(self.benchmark.dimension)], self.benchmark
            )
        FLC = self.config.flowstar_config
        abstraction.to_flowstar(out_file, self.config, FLC)


def node1():
    CONFIG_FILE = "experiments/node1/gen-config.yaml"
    config = cli.get_config_from_yaml(CONFIG_FILE)
    synth = Synth(config)
    node = synth.create_from_benchmark()
    synth.learner_flowstar(node, "experiments/node1/node1")
    torch.save(node, "experiments/node1/node1.pt")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    node1()
