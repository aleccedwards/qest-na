# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
from typing import List, OrderedDict

import numpy as np
import pyswarms as ps
import torch
from matplotlib import pyplot as plt
from scipy import optimize as opt
from torch import Tensor, nn, optim

from utils import Timer, timer

# import neural_abstraction as na

T = Timer()
best_loc = None


def vprint(m, v: bool):
    """Print first argument if second argument is True."""
    if v:
        print(m)


class ReCU(nn.Module):
    """Step function activation function"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.where(x >= 0, torch.ones_like(x), torch.zeros_like(x))


class LeakyReCU(nn.Module):
    """Step function activation function with linear leaky slope"""

    def __init__(self) -> None:
        self.a = 0.05
        super().__init__()

    def forward(self, x):
        return torch.where(
            x >= 0, torch.ones_like(x) + self.a * x, torch.zeros_like(x) + self.a * x
        )


class ReluNet(nn.Module):
    """torch network of arbitrary size with ReLU activation functions

    Args:
        nn (Module): inherits from torch.nn.module
    """

    def __init__(self, benchmark, width: List[int], error: float, config) -> None:
        """Initialise ReLU net object

        Args:
            benchmark (Benchmark): Benchmark object describing model to be abstracted
            width (List[int]): size of hidden layers of network
            error (float): target error (potentially redundant)
            config (_type_): configuration object of program
        """
        super().__init__()
        self.width = width
        self.benchmark = benchmark
        self.loss = nn.MSELoss(reduction="mean")
        self.relu = nn.ReLU
        self.error = torch.tensor(error) if error else error
        self.model = nn.Sequential(self.get_structure(width))
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        """forward method for nn.module"""
        return self.model(x)

    @timer(T)
    def learn(self, S: Tensor, y: Tensor, optimizer: optim.Optimizer, scheduler=None):
        """Trains neural abstraction.

        Trains neural abstraction using the corresponding method described by
        the configuration object.

        Args:
            S (Tensor): Training Data
            y (Tensor): _description_
            optimizer (optim.Optimizer): _description_

        """
        if self.config.learning_mode == "error":
            return self.learn_error(S, y, optimizer, scheduler)
        elif self.config.learning_mode == "PSO":
            return self.learn_PSO(S, y)
        elif self.config.learning_mode == "DE":
            return self.learn_DE(S, y)
        else:
            return self.learn_min(S, y, optimizer, scheduler)

    def regularisation_loss(self, l):
        W = self.model[0].weight
        b = self.model[0].bias
        cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(W.shape[0] - 1):
            row = W[i, :]
            for rowj, bj in zip(W[1 + i :], b[1 + i :]):
                q = cosine(row, rowj) ** 2
                q2 = (b[i] - bj) ** 2
                l = l + 1 / (q * q2)
        return l

    def prune(self):
        W = self.model[0].weight
        b = self.model[0].bias
        cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        remove = []
        for i in range(W.shape[0] - 1):
            row = W[i, :]
            for rowj, bj in zip(W[1 + i :], b[1 + i :]):
                q = cosine(row, rowj) ** 2
                q2 = (b[i] - bj) ** 2
                if q < 0.01 and q2 < 0.01:
                    remove.append(i)
        print(len(remove))
        for i in remove:
            with torch.no_grad():
                self.model[0].weight[i] = torch.nn.Parameter(
                    torch.zeros_like(self.model[0].weight[i])
                )
                self.model[0].bias[i] = torch.nn.Parameter(
                    torch.zeros_like(self.model[0].bias[i])
                )

    def learn_min(
        self, S: Tensor, y: Tensor, optimizer: optim.Optimizer, scheduler=None
    ) -> None:
        """Trains neural abstraction to a target local minima.

        Performs gradient descent using the optimizer parameter, using stopping criterion from the self.config object.

        Args:
            S (Tensor): Training Data
            y (Tensor): Target Data
            optimizer (optim.Optimizer): Torch optimizer object

        """
        stop = False
        # grad_prev = float("inf")
        interior_points = self.benchmark.domain.check_interior(S)
        # grad = []
        # grad_grad = []
        l = []
        i = 0
        while not stop:
            # s = 0
            i += 1
            loss = self.loss(self(S), y)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            e = abs((self(S) - y)[interior_points]).max(axis=0)[0]

            # for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            #     s += p.grad.data.norm(2).item()
            # grad.append(s)
            # d_grad = s - grad_prev
            # grad_grad.append(d_grad)
            # grad_prev = s
            l.append(loss.item())
            optimizer.zero_grad()
            if i % 100 == 0:
                vprint(loss.item(), self.config.verbose)
            if (
                loss.item()
                < self.config.loss_stop
                # and s < self.config.loss_grad_stop
                # and d_grad < self.config.loss_grad_grad_stop
            ):
                stop = True
        e = abs((self(S) - y)[interior_points]).max(axis=0)[0]
        e = (e / 0.8).tolist()
        e = [round(max(ei, 0.005), ndigits=3) for ei in e]
        return e

    def learn_error(
        self, S: Tensor, y: Tensor, optimizer: optim.Optimizer, scheduler=None
    ) -> None:
        """Trains neural abstraction to a target maximum error threshold.

        Performs gradient descent using the optimizer parameter, using error based stopping criterion from the self.config object.

        Args:
            S (Tensor): Training Data
            y (Tensor): Target Data
            optimizer (optim.Optimizer):  Torch optimizer object
        """
        n_error = float("inf")
        split = int(0.9 * S.shape[0])
        S_train, S_val = S[:split], S[split:]
        y_train, y_val = y[:split], y[split:]

        iter = 0
        interior_points_train = self.benchmark.domain.check_interior(S_train)
        # interior_points_val = self.benchmark.domain.check_interior(S_val)
        while n_error > 0:
            iter += 1
            loss = self.loss(self(S), y)
            error = (
                ((self(S_train) - y_train).abs() > 0.80 * self.error)[
                    interior_points_train
                ]
                .any(dim=1)
                .sum()
            )

            n_error = error.sum().item()

            if iter % 100 == 0:
                vprint(f"ME = {loss}, N_error = {n_error}", self.config.verbose)

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    def learn_PSO(self, S: Tensor, y: Tensor):
        """Trains a neural network using the PSO algorithm.

        PSO is performed using PySwarms.

        TODO: Add settings for PSO to function call.

        Args:
            S (Tensor): Training Data
            y (Tensor): Target Data

        Returns:
            error (list): list of achieved maximum error for each dimension
        """
        global best_loc
        interior_points = self.benchmark.domain.check_interior(S)
        S, y = S.numpy(), y.numpy()
        flatnet = FlatNet(self, S, y)
        options = {"c1": 0.9, "c2": 0.3, "w": 0.9}
        # Call instance of PSO
        dimensions = flatnet.np
        NP = 20
        optimizer = ps.single.GlobalBestPSO(
            n_particles=NP,
            dimensions=dimensions,
            options=options,
            ftol_iter=5,
            init_pos=best_loc,
        )

        # Perform optimization
        cost, pos = optimizer.optimize(flatnet.f, iters=1000, verbose=False)
        best_loc = np.tile(pos, (NP, 1))
        x = flatnet.update_torch_net(pos)
        self.__dict__.update(x.__dict__)
        S, y = torch.tensor(S).float(), torch.tensor(y).float()
        e = (abs(self(S) - y))[interior_points].max(axis=0)[0]
        e = [ei / 0.8 for ei in e.tolist()]
        return e

    def learn_DE(self, S: Tensor, y: Tensor):
        """Trains a neural network using the DE algorithm.

        DE is performed using Scipy.

        TODO: Add settings for DE to function call.

        Args:
            S (Tensor): Training Data
            y (Tensor): Target Data

        Returns:
            error (list): list of achieved maximum error for each dimension
        """
        np.random.seed(0)
        interior_points = self.benchmark.domain.check_interior(S)
        S, y = S.numpy(), y.numpy()
        flatnet = FlatNet(self, S, y)

        bounds = [[-5, 5] for i in range(flatnet.np)]
        res = opt.differential_evolution(
            flatnet.f_de,
            bounds,
            updating="deferred",
            disp=True,
            vectorized=True,
            popsize=10,
            maxiter=300,
            strategy="best1exp",
        )
        x = flatnet.update_torch_net(res.x)
        self.__dict__.update(x.__dict__)
        S, y = torch.tensor(S).float(), torch.tensor(y).float()
        e = (abs(self(S) - y))[interior_points].max(axis=0)[0]
        e = [ei / 0.8 for ei in e.tolist()]
        return e

    def get_structure(self, width: List[int]) -> OrderedDict:
        """returns ordered dictionary of hidden layers with relu activations based on width list

        Args:
            width (List[int]): size of hidden layers of net

        Returns:
            OrderedDict: (label: layer) pairs representing neural network structure in order.
        """
        input_layer = [
            (
                "linear-in",
                nn.Linear(
                    self.benchmark.dimension,
                    width[0],
                ),
            )
        ]
        out = [
            (
                "linear-out",
                nn.Linear(
                    width[-1],
                    self.benchmark.dimension,
                ),
            )
        ]
        relus = [("relu" + str(i + 1), self.relu()) for i in range(len(width))]
        lins = [
            ("linar" + str(i + 1), nn.Linear(width[i], width[i + 1]))
            for i in range(len(width) - 1)
        ]
        z = [None] * (2 * len(width) - 1)
        z[::2] = relus
        z[1::2] = lins
        structure = OrderedDict(input_layer + z + out)
        return structure

    @staticmethod
    def get_timer():
        return T


class ReconstructedRelu(nn.Module):
    """Reconstructs ScalarReluNet networks into a single vector-valued function.

    Args:
        nn (nn.module): inherits from torch.nn.Module
    """

    def __init__(
        self,
        scalar_nets,
    ) -> None:
        super().__init__()
        self.scalar_nets = scalar_nets

    def forward(self, x):
        return (
            torch.stack([self.scalar_nets[i](x) for i in range(len(self.scalar_nets))])
            .squeeze()
            .T
        )


class PWCNet(ReluNet):
    """Variant of ReluNet that uses a piecewise constant activation function."""

    def __init__(self, benchmark, width: List[int], error: float, config) -> None:
        super().__init__(benchmark, width, error, config)
        self.relu = ReCU
        self.model = nn.Sequential(self.get_structure(width))


class SigmoidNet(ReluNet):
    """Variant of ReluNet that uses a sigmoid activation function."""

    def __init__(self, benchmark, width: List[int], error: float, config) -> None:
        super().__init__(benchmark, width, error, config)
        self.relu = nn.Sigmoid
        self.model = nn.Sequential(self.get_structure(width))


class TanhNet(ReluNet):
    def __init__(self, benchmark, width: List[int], error: float, config) -> None:
        super().__init__(benchmark, width, error, config)
        self.relu = nn.Tanh
        self.model = nn.Sequential(self.get_structure(width))


class FlatNet:
    """Object that represents a torch network parameters using a single 1D vector.

    FlatNet represents a ReLUNet, but the
    weights W and biases b are represented
    as a single 1D vector.
    """

    def __init__(self, relunet: ReluNet, S, y):
        """intializes FlatNet object.

        Args:
            relunet (ReluNet): RelUNet object to be represented
            S (_type_): Training data
            y (_type_): Target data
        """
        self.relunet = relunet
        self.p = self.get_n_params()
        self.np = sum([sum([torch.prod(pi[0]).item(), pi[1].item()]) for pi in self.p])
        self.in_vars = self.p[0][0][1].item()
        self.out_vars = self.p[-1][0][0].item()
        self.S = S
        self.y = y
        assert self.in_vars == self.out_vars

    def get_n_params(self):
        """Get shape of parameters in the ReluNet.

        Returns:
            p (list): list of (Weight.shape, bias.shape) pairs
        """
        p = []
        for layer in self.relunet.model:
            if not isinstance(layer, torch.nn.Linear):
                continue
            W, b = torch.tensor(layer.weight.shape), torch.tensor(layer.bias.shape)
            p.append((W, b))
        return p

    def forward(self, theta):
        """forward pass of FlatNet.
        Args:
            theta (np.array):
                vector of all network parameters (weights & biases).

        Returns:
            z: output of network.
        """
        cum_i = 0
        z = self.S.T
        for layer in self.p[:-2]:
            Wi_shape = layer[0]
            bi_shape = layer[1]
            Wi = theta[
                cum_i : cum_i + (Wi_shape[0].item() * Wi_shape[1].item())
            ].reshape(Wi_shape.numpy())
            cum_i += Wi_shape[0].item() * Wi_shape[1].item()
            bi = (
                theta[cum_i : cum_i + bi_shape.item()]
                .reshape(bi_shape.numpy())
                .reshape(-1, 1)
            )
            cum_i += bi_shape.item()
            z = Wi @ z + bi
            z = np.maximum(z, 0)

        ## Penultimate layer (with pwc output)
        layer = self.p[-2]
        Wi_shape = layer[0]
        bi_shape = layer[1]
        Wi = theta[cum_i : cum_i + (Wi_shape[0].item() * Wi_shape[1].item())].reshape(
            Wi_shape.numpy()
        )
        cum_i += Wi_shape[0].item() * Wi_shape[1].item()
        bi = (
            theta[cum_i : cum_i + bi_shape.item()]
            .reshape(bi_shape.numpy())
            .reshape(-1, 1)
        )
        cum_i += bi_shape.item()
        z = Wi @ z + bi
        # z = np.maximum(z, 0)
        z = np.heaviside(z, z)

        # Final layer (linear transformation of pwc output)
        layer = self.p[-1]
        Wi_shape = layer[0]
        bi_shape = layer[1]
        Wi = theta[cum_i : cum_i + (Wi_shape[0].item() * Wi_shape[1].item())].reshape(
            Wi_shape.numpy()
        )
        cum_i += Wi_shape[0].item() * Wi_shape[1].item()
        bi = (
            theta[cum_i : cum_i + bi_shape.item()]
            .reshape(bi_shape.numpy())
            .reshape(-1, 1)
        )
        cum_i += bi_shape.item()
        z = Wi @ z + bi
        return z

    def loss(self, theta):
        """Calculates MSE loss of FlatNet in terms of parameters.

        Args:
            theta (np.ndarray): flattened vector of all network parameters.

        Returns:
            loss (np.ndarray): Symmetric MSE loss of FlatNet.
        """
        sdot = self.forward(theta)
        loss = np.max(
            np.linalg.norm(sdot.T - self.y, axis=0, ord=np.inf)
        )  # + np.mean(np.linalg.norm(sdot.T - self.y, axis=1))
        return loss

    def f(self, x):
        """Higher-level method to do forward_prop in the
        whole swarm.

        Args:
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        j = [self.loss(x[i]) for i in range(n_particles)]
        return np.array(j)

    def f_de(self, x):
        """Higher-level method to do forward_prop in the
        whole swarm.

        Args:
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        try:
            n_particles = x.shape[1]
        except IndexError:
            n_particles = 1
        if n_particles > 1:
            j = [self.loss(x[:, i]) for i in range(n_particles)]
        else:
            j = [self.loss(x)]
        return np.array(j)

    def update_torch_net(self, pos):
        """Update ReluNet with new weights and biases from pos.

        Args:
            pos (np.ndarray):
                Flat representation of the network parameters from the optimizer.

        Returns:
            self.relunet (ReLUNet):
                ReLU Network with updated weights and biases.
        """
        cum_i = 0
        with torch.no_grad():
            for i, layer in enumerate(self.p):
                Wi_shape = layer[0]
                bi_shape = layer[1]
                Wi = pos[
                    cum_i : cum_i + (Wi_shape[0].item() * Wi_shape[1].item())
                ].reshape(Wi_shape.tolist())
                cum_i += Wi_shape[0].item() * Wi_shape[1].item()
                bi = pos[cum_i : cum_i + bi_shape.item()].reshape(bi_shape.tolist())
                cum_i += bi_shape.item()
                self.relunet.model[2 * i].weight = torch.nn.parameter.Parameter(
                    torch.tensor(Wi).float()
                )
                self.relunet.model[2 * i].bias = torch.nn.parameter.Parameter(
                    torch.tensor(bi).float()
                )
        return self.relunet


if __name__ == "__main__":
    from scipy import optimize as opt

    import benchmarks
    from cli import get_config

    c = get_config()
    b = benchmarks.read_benchmark("lin")
    net = ReluNet(b, [40], 0, c)
    S = b.get_data()
    y = torch.tensor(list(map(b.f, S)))
    S, y = S.numpy(), y.numpy()
    fnet = FlatNet(net, S, y)
    options = {"c1": 0.7, "c2": 0.3, "w": 0.9}

    # Call instance of PSO
    dimensions = fnet.np
    optimizer = ps.single.GlobalBestPSO(
        n_particles=20, dimensions=dimensions, options=options
    )

    # Perform optimization
    cost, pos = optimizer.optimize(fnet.f, iters=500)
    print(cost)
    # x = fnet.update_torch_net(pos)
    # bounds = [[-5, 5] for i in range(fnet.np)]
    # res = opt.differential_evolution(fnet.loss, bounds, maxiter=10)
    # pass
    # x = 1
