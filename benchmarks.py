# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


from typing import Callable, List, Union

import numpy as np
import sympy as sp
import torch

import cegis.translator as translator
import cegis.verifier as ver
import sysplot
from domains import Rectangle, Sphere


class Benchmark:
    """Non-linear dynamical model"""

    def __init__(self) -> None:
        self.dimension: int = None
        self.name: str = None
        self.domain: Union[Rectangle, Sphere] = None

    def f(self, v):
        """function to evaluate dynamic model at given point

        Evaluates (x_0, ..., x_n) -> f(x_0, ..., x_n-1)

        Args:
            v (List): list of variables (x_0, ..., x_n)

        Returns:
            List: vector of values of f(x_0, ..., x_n-1)
        """
        try:
            f = self.f_num(v)
            if isinstance(v, np.ndarray):
                f = f.detach().numpy()
        except (TypeError, RuntimeError):
            f = self.f_sym(v)
        return f

    def get_domain(self, x: List, _And: Callable):
        """Returns symbolic formula for domain in terms of x.

        Args:
            x (List): symbolic variables.
            _And (Callable): And function for symbolic formula.

        Returns:
            domain: symbolic formula for domain.
        """
        return self.domain.generate_domain(x, _And)

    def get_data(self, n=10000):
        """Returns data points uniformly distributed over a slightly larger rectangle.

        Args:
            batch_size (int): nuber of points to sample.
            bloat (float, optional):  additive increase in size of rectangle. Defaults to 0.1.

        Returns:
            torch.Tensor: sampled data
        """
        return self.domain.generate_bloated_data(n)

    def plotting(self, net, name: str = None):
        """Plots the benchmark and a neural network's output.

        Args:
            net (ReLUNet): Neural network to plot.
            name (str, optional):
                Name of file to save plot to. If None, then plot is show.
                Defaults to None.
        """
        sysplot.plot_vector_fields(net, self, [-1, 1], [-1, 1], name=name)

    def get_funcs(self, v):
        if ver.DRealVerifier.check_type(v):
            return ver.DRealVerifier.solver_fncts()
        elif ver.Z3Verifier.check_type(v):
            return ver.Z3Verifier.solver_fncts()
        else:
            return {
                "cos": sp.cos,
                "sin": sp.sin,
                "exp": sp.exp,
                "log": sp.log,
                "sqrt": sp.sqrt,
                "pow": sp.Pow,
                "abs": abs,
            }


class Linear(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Linear"
        self.short_name = "lin"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1, 1]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        f = [-x + y, -y]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y = v[0], v[1]
        f = [-x + y, -y]
        return f

    def get_domain(self, x: List, _And):
        """
        Returns smt (symbolic) domain object
        """
        return self.domain.generate_domain(x, _And)

    def get_data(self, n=10000):
        """
        Returns tensor of data points sampled from domain
        """
        return self.domain.generate_bloated_data(n)

    def plotting(self, net, name: str = None):
        sysplot.plot_vector_fields(net, self, [-1, 1], [-1, 1], name=name)


class Linear3D(Benchmark):
    def __init__(self) -> None:
        self.dimension = 3
        self.name = "Linear3D"
        self.short_name = "lin"
        self.domain = Rectangle([-1, -1, -1], [1, 1, 1])
        self.scale = [1, 1, 1]

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        f = [-x + y, -y, -z]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y, z = v[0], v[1], v[2]
        f = [-x + y, -y, -z]
        return f


class NL1(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Non-Lipschitz1"
        self.short_name = "NL1"
        self.domain = Rectangle([0, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def get_data(self, n=10000):
        """
        Returns tensor of data points sampled from domain
        """
        return self.domain.generate_bloated_data(n, bloat=0)

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        f = [y, torch.sqrt(x)]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _sqrt = fncs["sqrt"]
        x, y = v[0], v[1]
        f = [y, _sqrt(x)]
        return f


class NL2(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Non-Lipschitz2"
        self.short_name = "NL2"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        f = [x**2 + y, torch.pow(torch.pow(x, 2), 1 / 3) - x]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _pow = fncs["pow"]
        x, y = v[0], v[1]
        f = [x**2 + y, _pow(_pow(x, 2), 1 / 3) - x]
        return f


class WaterTank(Benchmark):
    def __init__(self) -> None:
        self.dimension = 1
        self.name = "Water-tank"
        self.short_name = "tank"
        self.domain = Rectangle([0], [2])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def get_data(self, n=10000):
        """
        Returns tensor of data points sampled from domain
        """
        return self.domain.generate_bloated_data(n, bloat=0)

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x = v
        f = [1.5 - np.sqrt(x)]
        return torch.stack(f).reshape(v.shape)

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _sqrt = fncs["sqrt"]
        x = v[0]
        f = [1.5 - _sqrt(x)]
        return f

    def plotting(self, net, name: str = None):
        return


class WaterTankND(Benchmark):
    def __init__(self) -> None:
        self.name = "Water-tank-{}d".format(self.dimension)
        self.short_name = "tank{}d".format(self.dimension)
        self.domain = Rectangle([0] * self.dimension, [1] * self.dimension)
        self.scale = [1 for i in range(self.dimension)]

    def get_data(self, n=10000):
        """
        Returns tensor of data points sampled from domain
        """
        D = self.domain.generate_bloated_data(n)
        filt = D >= 0
        filt = filt[:, 0]  # Filter x0 only
        D = D[filt]
        return D

    def f(self, v):
        try:
            f = self.f_num(v)
            if isinstance(v, np.ndarray):
                f = f.detach().numpy()
        except (TypeError, RuntimeError):
            f = self.f_sym(v)
        return f

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x0 = v[:, 0]
        x1 = v[:, 1]
        a = [-v[:, i] for i in range(1, self.dimension - 1)]
        # a.reverse()
        c = 1 / self.dimension
        f = [0.2 - torch.sqrt(x0), *a, -c * (v.sum(dim=1))]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _sqrt = fncs["sqrt"]
        x0, x1 = v[0], v[1]
        a = [-v[i] for i in range(1, self.dimension - 1)]
        # a.reverse()
        c = 1 / self.dimension
        f = [0.2 - _sqrt(x0), *a, -c * (sum(v))]
        return f


class WaterTank2D(WaterTankND):
    def __init__(self) -> None:
        self.dimension = 2
        super().__init__()


class WaterTank3D(WaterTankND):
    def __init__(self) -> None:
        self.dimension = 3
        super().__init__()


class WaterTank4D(WaterTankND):
    def __init__(self) -> None:
        self.dimension = 4
        super().__init__()


class WaterTank5D(WaterTankND):
    def __init__(self) -> None:
        self.dimension = 5
        super().__init__()


class WaterTank6D(WaterTankND):
    def __init__(self) -> None:
        self.dimension = 6
        super().__init__()


class NODEBenchmark(Benchmark):
    def __init__(self):
        self.net = torch.load(self.MODEL_FILE)
        self.net.eval()
        if isinstance(self.net.model[1], torch.nn.Tanh):
            self.sigma = ver.DRealVerifier.tanh
        else:
            self.sigma = ver.DRealVerifier.sigmoid
        # Assume tanh activation, TODO: make this more general

    def f(self, v):
        """function to evaluate dynamic model at given point

        Evaluates (x_0, ..., x_n) -> f(x_0, ..., x_n-1)

        Args:
            v (List): list of variables (x_0, ..., x_n)

        Returns:
            List: vector of values of f(x_0, ..., x_n-1)
        """
        try:
            f = self.f_num(v)
            if isinstance(v, np.ndarray):
                f = f.detach().numpy()
        except (TypeError, RuntimeError):
            f = self.f_sym(v)
        return f

    def f_num(self, v):
        with torch.no_grad():
            if not torch.is_tensor(v):
                v = torch.tensor(v)
                v = v.type(torch.float32)
            return self.net(v)

    def f_sym(self, v):
        trans = translator.Translator(
            v, sigma_hidden=self.sigma, sigma_final=self.sigma
        )
        sym_net = trans.translate(self.net)
        return sym_net


class NODE1(NODEBenchmark):
    def __init__(self):
        self.MODEL_FILE = "experiments/node1/node1.pt"
        self.name = "NODE1"
        self.short_name = "node1"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.dimension = 2
        self.scale = [1 for i in range(self.dimension)]
        super().__init__()


def read_benchmark(name: str):
    """Reads a benchmark from a string and returns a Benchmark object

    Args:
        name (str): corresponding to a benchmark name

    Returns:
        benchmark (Benchmark): benchmark object
    """
    if name == "lin":
        return Linear()
    if name == "lin3d":
        return Linear3D()
    elif name == "nl1":
        return NL1()
    elif name == "nl2":
        return NL2()
    elif name == "tank":
        return WaterTank()
    elif name == "tank2":
        return WaterTank2D()
    elif name == "tank3":
        return WaterTank3D()
    elif name == "tank4":
        return WaterTank4D()
    elif name == "tank5":
        return WaterTank5D()
    elif name == "tank6":
        return WaterTank6D()
    elif name == "node1":
        return NODE1()
    else:
        raise ValueError("Benchmark {} not found".format(name))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--benchmark", type=str, help="benchmark name", default="nl2"
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="filename to save to",
        default=None,
        required=False,
    )
    args = parser.parse_args()
    b = read_benchmark(args.benchmark)
    if b.dimension == 2:
        from matplotlib import pyplot as plt

        sysplot.plot_benchmark(b)
        if args.filename is not None:
            plot = plt.gca()
            plt.savefig(args.filename)
        else:
            sysplot.show()
