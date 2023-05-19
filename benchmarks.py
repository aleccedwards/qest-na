# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


from typing import Callable, List, Union

import numpy as np
import sympy as sp
import torch
from interval import imath

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

    def get_image(self):
        """Find the image of the function wrt the domain."""
        if isinstance(self.domain, Sphere):
            raise ValueError("Intervals not implemented for spherical domains")
        return self.f(self.domain.as_intervals())

    def normalise(self):
        """Normalise dynamics to [-1, 1]^d."""
        scale = []
        for dim in self.image:
            scale.append((dim[0].sup - dim[0].inf) / 2)
        self.scale = scale
        self.name += "-normalised"

    def unnormalise(self):
        """Unnormalise dynamics"""
        self.scale = [1 for i in range(self.dimension)]
        self.name = self.name[:-11]

    def get_scaling(self):
        """Determine scaling for normalisation.

        Returns:
            shift (np.ndarray): shift for normalisation.
            scale (np.ndarray): scale for normalisation.
        """
        scales = []
        shifts = []
        for dim in self.image:
            shifts.append(dim.midpoint[0].inf)
            scales.append((dim[0].sup - dim[0].inf) / 2)
        return np.array(shifts).reshape(-1, 1), list(scales)

    def f_intervals(self, v):
        """Evaluate model using interval arithmetic.

        Relies on PyInterval."""
        return self.f(v)

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


class LoktaVolterra(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "lokta-volterra"
        self.short_name = "lv"
        self.alpha, self.beta, self.delta, self.gamma = [0.6, 1, 1, 0.6]
        self.domain = Rectangle([0, 0], [1, 1])
        self.scale = [1, 1]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        f = [self.alpha * x - self.beta * x * y, self.delta * x * y - self.gamma * y]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y = v[0], v[1]
        f = [self.alpha * x - self.beta * x * y, self.delta * x * y - self.gamma * y]
        return f

    def get_domain(self, x: List, _And):
        return self.domain.generate_domain(x, _And)

    def get_data(self, n=10000):
        return self.domain.generate_bloated_data(n)

    def plotting(self, net, name: str = None):
        sysplot.plot_vector_fields(net, self, [0, 1], [0, 1], name=name)


class Parillo1(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Parillo1"
        self.short_name = "p1"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        x = x * 2
        y = y * 2
        f = [-x + x * y, -y]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y = v[0], v[1]
        x = x * 2
        y = y * 2
        f = [-x + x * y, -y]
        return f

    def get_domain(self, x: List, _And):
        return self.domain.generate_domain(x, _And)

    def get_data(self, n=1000):
        return self.domain.generate_bloated_data(n)

    def plotting(self, net, name: str = None):
        sysplot.plot_vector_fields(net, self, [-1, 1], [-1, 1], name=name)


class VanderPol(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "VdP"
        self.short_name = "vdp"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        mu = 0.707
        x = x * 3
        y = y * 3
        f = [y, (mu * (1 - x**2) * y - x)]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y = v[0], v[1]
        mu = 0.707
        x = x * 3
        y = y * 3
        f = [y, (mu * (1 - x**2) * y - x)]
        return f

    def get_data(self, n=10000):
        return self.domain.generate_bloated_data(n, bloat=0.2)


class VanderPol2(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "vdp2"
        self.short_name = "vdp2"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        mu = 0.707
        c = 1.5
        x = x * 2
        y = y * 2
        f = [y, (mu * (1 - x**2) * y - x)]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y = v[0], v[1]
        mu = 0.707
        c = 1.5
        x = x * 2
        y = y * 2
        f = [y, (mu * (1 - x**2) * y - x)]
        return f

    def get_data(self, n=10000):
        return self.domain.generate_bloated_data(n, bloat=0.2)


class NP3(Benchmark):
    def __init__(self) -> None:
        self.dimension = 3
        self.name = "NP3"
        self.short_name = "np3"
        self.domain = Rectangle([-1, -1, -1], [1, 1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y, z = v
        f = [-3 * x - 0.1 * x * y**3, -y + z, -z]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y, z = v
        f = [-3 * x - 0.1 * x * y**3, -y + z, -z]
        return f


class ExponentialOld(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "ExponentialOld"
        self.short_name = "exp_old"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        x = x * 4
        y = y * 4
        f = [torch.exp(-x) + y - 1, -(torch.sin(x) ** 2)]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _exp = fncs["exp"]
        _sin = fncs["sin"]
        x, y = v[0], v[1]
        f = [_exp(-x) + y - 1, -(_sin(x) ** 2)]
        return f

    def f_intervals(self, v):
        x, y = v
        x = x * 4
        y = y * 4
        f = [imath.exp(-x) + y - 1, imath.sin(x) ** 2]
        return f

    def get_image(self):
        return self.f_intervals(self.domain.as_intervals())


class SteamGovernor(Benchmark):
    def __init__(self) -> None:
        self.dimension = 3
        self.name = "Steam"
        self.short_name = "steam"
        self.domain = Rectangle([-1, -1, -1], [1, 1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        f = [
            y,
            z**2 * torch.sin(x) * torch.cos(x) - torch.sin(x) - 3 * y,
            -(torch.cos(x) - 1),
        ]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _cos = fncs["cos"]
        _sin = fncs["sin"]
        x, y, z = v[0], v[1], v[2]
        f = [
            y,
            z**2 * _sin(x) * _cos(x) - _sin(x) - 3 * y,
            -(_cos(x) - 1),
        ]
        return f

    def f_intervals(self, v):
        x, y, z = v
        f = [
            y,
            z**2 * imath.sin(x) * imath.cos(x) - imath.sin(x) - 3 * y,
            -(imath.cos(x) - 1),
        ]
        return f

    def get_image(self):
        return self.f_intervals(self.domain.as_intervals())


class SpringPendulum(Benchmark):
    def __init__(self) -> None:
        self.dimension = 4
        self.name = "spring-pendulum"
        self.domain = Rectangle([-1, -1, -1, -1], [1, 1, 1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        r, t, vr, vt = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        g = 9.81
        f = [
            vr,
            vt,
            r * vt**2 + g * torch.cos(t) - 2 * (r - 1),
            -(2 * vr * vt + g * torch.sin(t)) / (r),
        ]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _cos = fncs["exp"]
        _sin = fncs["sin"]
        g = 9.81
        r, t, vr, vt = v[0], v[1], v[2], v[3]
        f = [
            vr,
            vt,
            r * vt**2 + g * _cos(t) - 2 * (r - 1),
            -(2 * vr * vt + g * _sin(t)) / (r),
        ]
        return f

    def f_intervals(self, v):
        r, t, vr, vt = v
        g = 9.81
        f = [
            vr,
            vt,
            r * vt**2 + g * imath.cos(t) - 2 * (r - 1),
            -(2 * vr * vt + g * imath.sin(t)) / (r),
        ]
        return f

    def get_image(self):
        return self.f_intervals(self.domain.as_intervals())


class FourDLoktaVolterra(Benchmark):
    def __init__(self) -> None:
        self.dimension = 4
        self.name = "4DLV"
        self.short_name = "4dlv"
        self.domain = Rectangle([0, 0, 0, 0], [1, 1, 1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x0, x1, x2, x3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        x = [x0, x1, x2, x3]
        r = [1.0, 0.72, 1.53, 1.27]
        a = [
            [1.0, 1.09, 1.52, 0.0],
            [0.0, 1.0, 0.44, 1.36],
            [2.33, 0.0, 1.0, 0.47],
            [1.21, 0.51, 0.35, 1.0],
        ]
        f0 = r[0] * x0 * (1 - sum([x[i] * a[0][i] for i in range(len(x))]))
        f1 = r[1] * x1 * (1 - sum([x[i] * a[1][i] for i in range(len(x))]))
        f2 = r[2] * x2 * (1 - sum([x[i] * a[2][i] for i in range(len(x))]))
        f3 = r[3] * x3 * (1 - sum([x[i] * a[3][i] for i in range(len(x))]))
        f = [f0, f1, f2, f3]
        return torch.stack(f).T

    def f_sym(self, v):
        x0, x1, x2, x3 = v[0], v[1], v[2], v[3]
        x = [x0, x1, x2, x3]
        r = [1.0, 0.72, 1.53, 1.27]
        a = [
            [1.0, 1.09, 1.52, 0.0],
            [0.0, 1.0, 0.44, 1.36],
            [2.33, 0.0, 1.0, 0.47],
            [1.21, 0.51, 0.35, 1.0],
        ]
        f0 = r[0] * x0 * (1 - sum([x[i] * a[0][i] for i in range(len(x))]))
        f1 = r[1] * x1 * (1 - sum([x[i] * a[1][i] for i in range(len(x))]))
        f2 = r[2] * x2 * (1 - sum([x[i] * a[2][i] for i in range(len(x))]))
        f3 = r[3] * x3 * (1 - sum([x[i] * a[3][i] for i in range(len(x))]))
        f = [f0, f1, f2, f3]
        return f


class CoupledVdP(Benchmark):
    def __init__(self) -> None:
        self.dimension = 4
        self.name = "Coupled VdP"
        self.short_name = "cvdp"
        self.domain = Rectangle([-1, -1, -1, -1], [1, 1, 1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x1, y1, x2, y2 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        mu = 1
        f = [
            y1,
            mu * (1 - x1**2) * y1 - x1 + (x2 - x1),
            y2,
            mu * (1 - x2**2) * y2 - x2 + (x1 - x2),
        ]
        return torch.stack(f).T

    def f_sym(self, v):
        x1, y1, x2, y2 = v[0], v[1], v[2], v[3]
        mu = 1
        f = [
            y1,
            mu * (1 - x1**2) * y1 - x1 + (x2 - x1),
            y2,
            mu * (1 - x2**2) * y2 - x2 + (x1 - x2),
        ]
        return f


class NeuronModel(Benchmark):
    def __init__(self) -> None:
        self.dimension = 3
        self.name = "neural"
        self.domain = Rectangle([-1, -1, -1], [1, 1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        s, xr, a, b, c, d = 4, -8 / 5, 1, 3, 1, 5
        I = 0
        r = 1e-3
        f = [
            y - a * x**3 + b * x**2 - z + I,
            c - d * x**2 - y,
            r * (s * (x - xr - z)),
        ]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y, z = v[0], v[1], v[2]
        s, xr, a, b, c, d = 4, -8 / 5, 1, 3, 1, 5
        I = 0
        r = 1e-3
        f = [
            y - a * x**3 + b * x**2 - z + I,
            c - d * x**2 - y,
            r * (s * (x - xr - z)),
        ]
        return f


class Brusselator(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Brusselator"
        self.short_name = "brus"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        x = x * 2
        y = y * 2
        f = [1 + x**2 * y - 1.5 * x - x, 1.5 * x - x**2 * y]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y = v[0], v[1]
        x = x * 2
        y = y * 2
        f = [1 + x**2 * y - 1.5 * x - x, 1.5 * x - x**2 * y]
        return f


class JetEngine(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Jet Engine"
        self.short_name = "jet"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        f = [-y - 1.5 * x**2 - 0.5 * x**3 - 0.1, 3 * x - y]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y = v[0], v[1]
        f = [-y - 1.5 * x**2 - 0.5 * x**3 - 0.1, 3 * x - y]
        return f


class Buckling(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Buckling"
        self.short_name = "buck"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        x = x * 2
        y = y * 2
        f = [y, 2 * x - x**3 - 0.2 * y + 0.1]
        return torch.stack(f).T

    def f_sym(self, v):
        x, y = v[0], v[1]
        x = x * 2
        y = y * 2
        f = [y, 2 * x - x**3 - 0.2 * y + 0.1]
        return f


class BioModel(Benchmark):
    def __init__(self) -> None:
        self.dimension = 7
        self.name = "bio-model"
        self.short_name = "bio"
        self.domain = Rectangle([0] * 7, [1] * 7)
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x1, x2, x3, x4, x5, x6, x7 = (
            v[:, 0],
            v[:, 1],
            v[:, 2],
            v[:, 3],
            v[:, 4],
            v[:, 5],
            v[:, 6],
        )
        x1 = x1 * 2
        x2 = x2 * 2
        x3 = x3 * 2
        x4 = x4 * 2
        x5 = x5 * 2
        x6 = x6 * 2
        x7 = x7 * 2
        f = [
            -0.4 * x1 + 5 * x3 * x4,
            0.4 * x1 - x2,
            x2 - 5 * x3 * x4,
            5 * x5 * x6 - 5 * x3 * x4,
            -5 * x5 * x6 + 5 * x3 * x4,
            0.5 * x7 - 5 * x5 * x6,
            -0.5 * x7 + 5 * x5 * x6,
        ]
        return torch.stack(f).T

    def f_sym(self, v):
        x1, x2, x3, x4, x5, x6, x7 = (
            v[0],
            v[1],
            v[2],
            v[3],
            v[4],
            v[5],
            v[6],
        )
        x1 = x1 * 2
        x2 = x2 * 2
        x3 = x3 * 2
        x4 = x4 * 2
        x5 = x5 * 2
        x6 = x6 * 2
        x7 = x7 * 2
        f = [
            -0.4 * x1 + 5 * x3 * x4,
            0.4 * x1 - x2,
            x2 - 5 * x3 * x4,
            5 * x5 * x6 - 5 * x3 * x4,
            -5 * x5 * x6 + 5 * x3 * x4,
            0.5 * x7 - 5 * x5 * x6,
            -0.5 * x7 + 5 * x5 * x6,
        ]
        return f


class LogSin(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "logsin"
        self.short_name = "logsin"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        y = y * 3
        f = [torch.sin(y), torch.log(x + 2)]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _sin, _log = fncs["sin"], fncs["log"]
        x, y = v[0], v[1]
        x = x
        y = y * 3
        f = [_sin(y), _log(x + 2)]
        return f

    def f_intervals(self, v):
        x, y = v
        f = [
            imath.sin(y),
            imath.log(x + 2),
        ]
        return f


class Log(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "log"
        self.short_name = "log"
        self.domain = Rectangle([0, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def get_data(self, n=10000):
        data = self.domain.generate_bloated_data(n, bloat=0.1)
        data[data[:, 0] < 0, 0] = 0
        return data

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        f = [torch.sqrt(x) + y, torch.log(x + 2)]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _sqrt, _log = fncs["sqrt"], fncs["log"]
        x, y = v[0], v[1]
        f = [np.sqrt(x) + y, np.log(x + 2)]
        return f

    def f_intervals(self, v):
        x, y = v
        f = [
            imath.sqrt(x) + y,
            imath.log(x + 2),
        ]
        return f

    def plotting(self, net, name: str = None):
        sysplot.plot_vector_fields(net, self, [0, 1], [-1, 1], name=name)


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


class NL3(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Non-Lipschitz3"
        self.short_name = "NL3"
        self.domain = Rectangle([0, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        f = [y, torch.pow(x, 1 / 3) - x]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _pow = fncs["pow"]
        x, y = v[0], v[1]
        f = [y, _pow(x, 1 / 3) - 0.2 * x]
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


class Exponential(Benchmark):
    def __init__(self) -> None:
        self.dimension = 2
        self.name = "Exponential"
        self.short_name = "exp"
        self.domain = Rectangle([-1, -1], [1, 1])
        self.scale = [1 for i in range(self.dimension)]
        # self.image = self.get_image()
        # self.normalise()#

    def f_num(self, v):
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        x, y = v[:, 0], v[:, 1]
        x = x * 4
        y = y * 4
        f = [-torch.sin(torch.exp(y**3 + 1)) - y**2, -x]
        return torch.stack(f).T

    def f_sym(self, v):
        fncs = self.get_funcs(v)
        _exp = fncs["exp"]
        _sin = fncs["sin"]
        x, y = v[0], v[1]
        f = [-_sin(_exp(y**3 + 1)) - y**2, -x]
        return f

    def f_intervals(self, v):
        x, y = v
        f = [imath.sin(imath.exp(y)), imath.exp(imath.sin(x))]
        return f


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


class NODE3D(NODEBenchmark):
    def __init__(self):
        self.MODEL_FILE = "experiments/node3d/node3d.pt"
        self.name = "NODE3D"
        self.short_name = "node3d"
        self.domain = Rectangle([-1, -1, -1], [1, 1, 1])
        self.dimension = 3
        self.scale = [1 for i in range(self.dimension)]
        super().__init__()


class NODE2(NODEBenchmark):
    def __init__(self):
        self.MODEL_FILE = "experiments/node2/node2.pt"
        self.name = "NODE2"
        self.short_name = "node2"
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
    if name == "lv":
        return LoktaVolterra()
    elif name == "exp" or name == "Exponential":
        return Exponential()
    elif name == "steam":
        return SteamGovernor()
    elif name == "sp":
        return SpringPendulum()
    elif name == "cvdp" or name == "coupled-vdp":
        return CoupledVdP()
    elif name == "neuron":
        return NeuronModel()
    elif name == "vdp":
        return VanderPol()
    elif name == "vdp2":
        return VanderPol2()
    elif name == "brus":
        return Brusselator()
    elif name == "jet":
        return JetEngine()
    elif name == "buck":
        return Buckling()
    elif name == "bio":
        return BioModel()
    elif name == "log":
        return Log()
    elif name == "nl1":
        return NL1()
    elif name == "nl2":
        return NL2()
    elif name == "nl3":
        return NL3()
    elif name == "logsin":
        return LogSin()
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
    elif name == "node3d":
        return NODE3D()
    else:
        raise ValueError("Benchmark {} not found".format(name))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--benchmark", type=str, help="benchmark name", default="buck"
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
