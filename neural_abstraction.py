# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import annotations

import copy
import pickle
import queue
from math import copysign

import numpy as np
import torch
import z3
from matplotlib import pyplot as plt

import polyhedrons
import sysplot
from automaton import Mode
from cegis import nn, translator, verifier
from config import VerifierType
from export import FlowstarWriter, XMLWriter
from utils import Stats, Timer, timer

T = Timer()


def increment(act: list) -> bool:
    """Increments the activation vector

    Args:
        act (List):
            bitvector representing activation configuration

    Returns:
        bool:
            True if increment reaches list of zeros, else None
    """
    for i in range(len(act)):
        if act[i] == 0:
            act[i] = 1
            return False
        else:
            act[i] = 0
    return True


def get_domain_constraints(domain, ndim, nvars=None):
    """Returns a list of constraints for a domain

    Generates array corresponding to constraints for box shaped domain
    of form [b -A] where b is the bounds in each support direction
    given by A


    Args:
        domain (Rectangle): Hyperrectangle domain
        ndim (_type_): number of dimensions
        nvars (_type_, optional):
        number of unique variables (if different to nvars). Defaults to None.

    Returns:
        np.ndarray: array of constraints
    """
    if nvars == None:
        nvars = ndim
    domain_constr = []
    for i in range(ndim):
        domain_constr.append(
            [
                abs(domain.lower_bounds[i]),
                *[
                    copysign(1, -domain.lower_bounds[i]) if j == i else 0
                    for j in range(ndim)
                ],
            ]
        )
        domain_constr.append(
            [
                abs(domain.upper_bounds[i]),
                *[
                    copysign(1, -domain.upper_bounds[i]) if j == i else 0
                    for j in range(ndim)
                ],
            ]
        )
    domain_constr = np.array(domain_constr)
    full_array = np.zeros((domain_constr.shape[0], nvars + 1))
    full_array[:, 0 : ndim + 1] = domain_constr
    return full_array


def construct_empty_constraint_matrices(W, b):
    """Constructs empty matrices of correct size for contsrains

    Args:
        W (List[torch.Tensor]): List of weight matrices
        b (List[torch.Tensor]): List of bias vectors

    Returns:
        ineq (np.ndarray): inequality constraint matrix
        eq (np.ndarray): equality constraint matrix
    """
    n_dim = W[0].shape[1]
    n_vars = n_dim + sum([W[i].shape[0] for i in range(len(W) - 2)])
    n_ineq_constr = sum([W[i].shape[0] for i in range(len(W) - 1)])
    n_eq_constr = sum(W[i].shape[0] for i in range(len(W) - 2))
    ineq = np.zeros((n_ineq_constr, 1 + n_vars))
    if n_eq_constr > 0:
        eq = np.zeros((n_eq_constr, 1 + n_vars))
    else:
        eq = None
    return ineq, eq


def get_constraints(ineq, eq, W, b, act):
    """Returns constraint matrices for a given activation vector

    Args:
        ineq (np.ndarray): empty inequality constraint matrix
        eq (np.ndarray): empty equality constraint matrix
        W (List(np.ndarray)): weight matrices
        b (List(np.ndarray)): bias vectors
        act (List): activation configuration

    Returns:
        ineq (np.ndarray): inequality constraint matrix
        eq (np.ndarray): equality constraint matrix
    """
    k = len(W)
    h_tot = 0
    var_tot = 0
    for i in range(k - 1):
        # For all hidden layers (and not output layer)
        Wi, bi = W[i], b[i]
        hi = Wi.shape[0]  # Number of neurons in layer
        ni = Wi.shape[1]  # Number of vars in layer (or neurons in prev layer)
        activation_i = act[h_tot : h_tot + hi]
        diag_a = np.diag([2 * a - 1 for a in activation_i])
        # Net is Wx + b >= 0   (including diags to switch inactive sign)
        # Cdd wants [c -A] from Ax <= 0
        # Therefore c = b, -W = -A
        ineq[h_tot : h_tot + hi, 0] = diag_a @ bi
        ineq[h_tot : h_tot + hi, var_tot + 1 : var_tot + ni + 1] = diag_a @ Wi
        if eq is not None and i != (k - 2):
            diag_a = np.diag([a for a in activation_i])
            # Should be able to exclude eq constraints for final layer
            eq[h_tot : h_tot + hi, 0] = -diag_a @ bi
            eq[h_tot : h_tot + hi, var_tot + 1 : var_tot + ni + 1] = -diag_a @ Wi
            eq[h_tot : h_tot + hi, var_tot + ni + 1 : var_tot + ni + hi + 1] = np.eye(
                hi
            )

        h_tot += hi
        var_tot += ni

    return ineq, eq


def check_fixed_hyperplanes(n_dim: int, domain_constr, W1, b1):
    """Checks if hyperplanes are fixed in the domain

    Checks if each neuron is only ever active or inactive. If only
    one then it can be fixed and 'pruned'
    First layer neurons only

    Args:
        n_dim (int): number of dimensions
        domain_constr (np.ndarray): domain constraints
        W1 (np.ndarray): weight matrix for first layer
        b1 (np.ndarray): bias vector for first layer

    Returns:
        dict:
            dict of index: fix of indices of
            neurons that can be fixed and the corresponding
            mode - active (1) or inactive (0)
    """
    activation_on = [1] * W1.shape[0]
    domain_constr = domain_constr[:, 0 : n_dim + 1]
    A = np.zeros((W1.shape[0], W1.shape[1] + 1))
    diag_a = np.diag([2 * a - 1 for a in activation_on])
    A[:, 0] = diag_a @ (b1)
    A[:, 1:] = diag_a @ (W1)

    on_res = []
    for plane in A:
        constr = np.vstack((domain_constr, plane))
        P = polyhedrons.Polyhedron(constr)
        on_res.append(int(P.is_nonempty()))
    activation_off = [0] * W1.shape[0]
    A = np.zeros((W1.shape[0], W1.shape[1] + 1))
    diag_a = np.diag([2 * a - 1 for a in activation_off])
    A[:, 0] = diag_a @ (b1)
    A[:, 1:] = diag_a @ (W1)
    off_res = []
    for plane in A:
        constr = np.vstack((domain_constr, plane))
        P = polyhedrons.Polyhedron(constr)
        off_res.append(-int(P.is_nonempty()))
    # If both on&off are non-empty, we want zero as cannot fix plane.
    # Otherwise want to fix plane to on (1) or off (0)
    combine_res = [a + b for a, b in zip(on_res, off_res)]
    fixed_res = {i: val for i, val in enumerate(combine_res) if val != 0}
    fixed_res = {i: val if val == 1 else 0 for i, val in fixed_res.items()}
    return fixed_res


def check_activation_cdd_1l(ineq_constr, eq_constr, W, b, activation):
    W1 = W[0]
    b1 = b[0]
    n_constr = ineq_constr.shape[0] - 2 * (ineq_constr.shape[1] - 1)
    diag_a = np.diag([2 * a - 1 for a in activation])
    # Net is Wx + b >= 0   (including diags to switch inactive sign)
    # Cdd wants [c -A] from Ax <= 0
    # Therefore c = b, -W = -A
    ineq_constr[:n_constr, 0] = diag_a @ (b1)
    ineq_constr[:n_constr, 1:] = diag_a @ (W1)
    P = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
    return P.is_nonempty()


@timer(T)
def check_activation_cdd(ineq_constr, eq_constr, W, b, activation):
    """Checks if activation is valid using cdd solver

    Using cdd-lib to check if a given activation configuration is
    active by constructing the convex polyhedron and check if it is
    non-empty.
    Args:
        ineq_constr (np.ndarray): inequality constraint
        eq_constr (np.ndarray): equality constraint
        W (List(np.ndarray)): weight matrices
        b (List(np.ndarray)): bias vectors
        activation (List): activation configuration
    Returns:
        bool: True if nonempty, False if not
    """
    ineq_constr, eq_constr = get_constraints(ineq_constr, eq_constr, W, b, activation)
    P = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
    return P.is_nonempty()


def enumerate_configurations(ineq_constr, eq_constr, W, b, fixed_acts={}):
    """Returns all possible activation configurations by enumeration

    Finds all active configs by enumerating through all possible
    configurations and checking the corresponding LP is feasible.

    Args:
        ineq_constr (np.ndarray): inequality constraint matrix
        eq_constr (np.ndarray): equality constraint matrix
        W (List(np.ndarray)): list of weight matrices
        b (List(np.ndarray)): list of bias vectors
        fixed_acts (dict, optional):
            dict of fixed activations. k-v pairs neuron index
            and mode (1 on and 0 off). Defaults to {}.

    Returns:
        activations (list): list of activation configurations
    """
    n_acts = sum([W[i].shape[0] for i in range(len(W) - 1)])
    activation = [0] * (n_acts - len(fixed_acts))
    feasible_acts = []
    # NOTE: There's a slight bug with this. If all activations are fixed (Not sure if this should be possible,
    # then this function returns not legit activations, when there should be a single one (the full fixed one)).
    # The below code acts as an edge case check for this, but this might not be correct behaviour
    if activation == []:
        return [list(fixed_acts.values())]
    while True:
        full_activation = copy.deepcopy(activation)
        _ = [full_activation.insert(i, val) for i, val in fixed_acts.items()]
        if check_activation_cdd(ineq_constr, eq_constr, W, b, full_activation):
            feasible_acts.append(copy.deepcopy(full_activation))
        if increment(activation):
            break
    # print(T)
    return feasible_acts


def get_active_configurations(domain, n_dim, W, b, mode="allsat"):
    """Find all activation configurations using a given technique

    Args:
        domain (domain.Rectangle):
            Rectangle domain with lower & upper bounds
        n_dim (int): number of dimensions of f(x)
        W (List(np.ndarray)): list of weight matrices
        b (_type_): list of bias vectors
        mode (str, optional): mode to use. Defaults to 'tree'.

    Returns:
        List: list of activation configurations
    """
    ineq_constr, eq_constr = construct_empty_constraint_matrices(W, b)
    domain_constr = get_domain_constraints(
        domain, W[0].shape[1], nvars=(ineq_constr.shape[1] - 1)
    )
    # fixed_acts = check_fixed_hyperplanes(n_dim, domain_constr, W[0], b[0])
    # print(len(fixed_acts))
    fixed_acts = {}
    ineq_constr = np.concatenate([ineq_constr, domain_constr])
    if mode == "tree":
        tree = NeuronTree(W, b, domain)
        configs = tree.explore_tree()
    elif mode == "allsat":
        configs = all_sat(W, b, domain)
    elif mode == "enum":
        configs = enumerate_configurations(
            ineq_constr, eq_constr, W, b, fixed_acts=fixed_acts
        )

    return configs


def get_Wb(nets):
    """Returns combined weight matrices and bias vectors for ReluNets

    This function is needed for deprecated ScalarCegis synthesis"""
    W = []
    b = []
    for net in nets:
        Wi = []
        bi = []
        for layer in net.model:
            if isinstance(layer, torch.nn.Linear):
                Wi.append(layer.weight)
                bi.append(layer.bias)
        W.append(Wi)
        b.append(bi)
    W, b = list(map(list, zip(*W))), list(map(list, zip(*b)))
    W = [torch.block_diag(*Wi) if i != 0 else torch.cat(Wi) for i, Wi in enumerate(W)]
    b = [torch.cat(bi) for bi in b]

    for w1, w2 in zip(W, W[1:]):
        assert w1.shape[0] == w2.shape[1]
    for w1, b1 in zip(W, b):
        assert w1.shape[0] == b1.shape[0]
    W = [Wi.detach().numpy().round(7) for Wi in W]
    b = [bi.detach().numpy().round(7) for bi in b]
    return W, b


def get_mode_flow(W, b, act):
    k = len(W)
    A = np.eye(W[0].shape[1])
    c = 0
    h_tot = 0
    layer_acts = []
    for i in range(k - 1):
        Wi = W[i]
        hi = Wi.shape[0]  # Number of neurons in layer
        activation_i = act[h_tot : h_tot + hi]
        layer_acts.append(activation_i)
        h_tot += hi

    for i in range(k - 1):
        Wi, bi = W[i], b[i]
        activation_i = layer_acts[i]
        diag_a = np.diag(activation_i)
        A = diag_a @ Wi @ A
        Wj = np.eye(W[i].shape[0])
        for j in range(i + 1, k - 1):
            diag_a = np.diag(layer_acts[j])
            Wj = diag_a @ W[j] @ Wj
        Wj = W[-1] @ Wj
        c += Wj @ np.diag(layer_acts[i]) @ bi
    A = W[-1] @ A
    c = b[-1] + c
    return A, c


def get_pwc_flow(W, b, act):
    # Tested for single layer only atm
    h_tot = 0
    layer_acts = []
    k = len(W)
    for i in range(k - 1):
        Wi = W[i]
        hi = Wi.shape[0]  # Number of neurons in layer
        activation_i = act[h_tot : h_tot + hi]
        layer_acts.append(activation_i)
        h_tot += hi

    act = layer_acts[-1]
    W0, b0 = W[0], b[0]
    W1, b1 = W[-1], b[-1]
    c = W1 @ np.array(act).reshape(-1, 1) + b1.reshape(-1, 1)
    A = np.zeros((c.shape[0], c.shape[0]))
    return A, c.squeeze(axis=1)


class NeuralAbstraction:
    """Class to represent a neural abstraction and determine the required
    objects to cast a ReluNet & error bound as a hybrid automaton."""

    def __init__(self, net, error, benchmark, template) -> None:
        """Initialise the NeuralAbstraction class for a given dynamical model

        Args:
            net (ReLUNet): trained neural network
            error (List(float)): error bound (for each dimension)
            benchmark (benchmarks.Benchmark): benchmark object
        """
        self.nets = net
        self.template = template
        self.reconstructed_net = nn.ReconstructedRelu(self.nets)
        self.dim = self.nets[0].model[0].weight.shape[1]
        self.error = error
        # print(self.error)
        self.benchmark = benchmark
        self.locations = self.get_activations()
        self.invariants = self.get_invariants()
        self.flows = self.get_flows()
        self.modes = self.get_modes()
        self.transitions = self.get_transitions()

    @timer(T)
    def get_activations(self) -> dict:
        """Get all valid activation configurations"""
        W, b = get_Wb(self.nets)
        acts = get_active_configurations(self.benchmark.domain, self.dim, W, b)
        return {str(i): acts[i] for i in range(len(acts))}

    @staticmethod
    def get_timer():
        return T

    def get_invariants(self) -> dict:
        """Get invariant polyhedrons for each activation configuration"""
        invariants = {}
        W, b = get_Wb(self.nets)
        ineq, eq = construct_empty_constraint_matrices(W, b)

        for loc, activation in self.locations.items():
            ineq_constr, eq_constr = get_constraints(ineq, eq, W, b, activation)
            domain_constr = get_domain_constraints(
                self.benchmark.domain, self.dim, nvars=(ineq_constr.shape[1] - 1)
            )
            ineq_constr = np.vstack([domain_constr, ineq_constr])
            P = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
            if len(self.nets[0].width) > 1:
                # print("Reducing to {} dimensions".format(self.dim))
                P.reduce_to_nd(self.dim)

            invariants.update({loc: P})
        return invariants

    def get_transitions(self) -> dict:
        """Detect all transitions between activation configurations"""
        transitions = []
        locations = list(self.locations.keys())
        for l0 in locations:
            P0 = self.invariants[l0]
            for l1 in locations:
                if l0 == l1:
                    # Self transition
                    continue
                P1 = self.invariants[l1]
                V = polyhedrons.get_shared_vertices(P0, P1, self.dim)
                if len(V) > 0:
                    transitions.append((l0, l1))
        return transitions

    def get_flows(self):
        """Get affine flow for each activation configuration"""
        flows = {}
        W, b = get_Wb(self.nets)
        for loc, act in self.locations.items():
            if self.template == "pwa":
                A, c = get_mode_flow(W, b, act)
            elif self.template == "pwc":
                A, c = get_pwc_flow(W, b, act)
            else:
                raise ValueError("Invalid template")
            flows.update({loc: (copy.deepcopy(A), copy.deepcopy(c))})
        return flows

    def get_modes(self) -> dict[str, Mode]:
        """Construct modes of abstraction"""
        modes = {}
        for loc in self.locations.keys():
            M = Mode(self.flows[loc], self.invariants[loc], self.error)
            modes.update({loc: M})
        return modes

    def find_mode(self, x: list):
        """Finds which mode a point lies in"""
        for loc, mode in self.modes.items():
            if mode.contains(x):
                return loc
        return None

    @timer(T)
    def prune_transitions_new(self, f_true=None, verbose=False, ver_type="dreal"):
        """Prune transitions that are never taken based on flow directions"""
        ver_type = (
            verifier.Z3Verifier
            if ver_type == VerifierType.Z3
            else verifier.DRealVerifier
        )
        dim = self.dim
        modes = self.modes
        transitions = self.transitions
        x = ver_type.new_vars(dim)
        ver = ver_type(x, dim, None)
        new_transitions = []
        solver = ver.new_solver()
        solver_funcs = ver.fncts
        _And = solver_funcs["And"]
        _Or = solver_funcs["Or"]
        for transition in transitions:
            source_id, destination_id = transition
            source = modes[source_id]
            destination = modes[destination_id]
            intersection = _And(
                source.P.as_smt(x, _And),
                destination.P.as_smt(x, _And),
            )
            source_flow_upper = source.flow_upper(np.array(x).reshape(1, -1))
            source_flow_lower = source.flow_lower(np.array(x).reshape(1, -1))

            lies_in = []
            for a, b in destination.P.hyperplanes():
                on_hyperplane = np.dot(a.reshape(1, -1), x).item() == b

                if f_true is not None:
                    f = np.array(f_true(x)).reshape(-1, 1)
                    lie = np.dot(a.reshape(1, -1), f).item()
                    f1 = _And(lie <= 0, on_hyperplane, intersection)
                    lies_in.append(f1)
                else:
                    lie_upper = np.dot(a.reshape(1, -1), source_flow_upper).item()
                    lie_lower = np.dot(a.reshape(1, -1), source_flow_lower).item()
                    f1 = _And(lie_lower <= 0, on_hyperplane, intersection)
                    f2 = _And(lie_upper <= 0, on_hyperplane, intersection)
                    lies_in.append(_Or(f1, f2))

            res = ver._solver_solve(solver, _Or(*lies_in))  # solver.add(_Or(lies_in))
            # res = solver.check()
            if ver.is_unsat(res):
                if verbose:
                    print("Transition pruned:", transition)
            else:
                new_transitions.append(transition)

            try:
                solver.reset()
            except AttributeError:
                pass

        r = (len(self.transitions) - len(new_transitions)) / len(self.transitions)
        if verbose:
            print("Pruned {}% of transitions".format(r * 100))

        self.transitions = new_transitions

    def plot(self, label=False, show=True, fname=None):
        """Plot the neural abstraction and its partitions"""
        net = nn.ReconstructedRelu(self.nets)
        domain = self.benchmark.domain
        xb = [domain.lower_bounds[0], domain.upper_bounds[0]]
        yb = [domain.lower_bounds[1], domain.upper_bounds[1]]
        sysplot.plot_nn_vector_field(net, xb, yb)
        for lab, inv in self.invariants.items():
            inv.plot(color="k")
            if label:
                c = np.array(inv.V)[:, 1:].mean(axis=0)
                plt.text(c[0], c[1], r"$P_{{{}}}$".format(lab), fontsize="large")
        plt.xticks([-1, 0, 1])
        plt.yticks([-1, 0, 1])
        plt.xlim(xb)
        plt.ylim(yb)
        plt.gca().set_aspect("equal")
        if show:
            plt.show()
        else:
            print("Saving to {}".format(fname))
            plt.savefig(fname + ".pdf", bbox_inches="tight")

    def error_plot(self):
        """Surface plot of the error of the abstraction"""
        domain = self.benchmark.domain
        xb = [domain.lower_bounds[0], domain.upper_bounds[0]]
        yb = [domain.lower_bounds[1], domain.upper_bounds[1]]
        x = np.linspace(xb[0], xb[1], 100)
        y = np.linspace(yb[0], yb[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        Z2 = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                p = np.array([x[i], y[j]]).reshape(1, -1)
                M_id = self.find_mode(p)
                M = self.modes[M_id]

                Z[i, j] = sum(M.disturbance)
                Z2[i, j] = sum(self.error)

        # surface plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(
            X, Y, Z, color="tab:blue", alpha=0.7, linewidth=0, label="Original Error"
        )
        ax.plot_surface(
            X, Y, Z2, color="tab:orange", linewidth=0, alpha=0.5, label="Improved Error"
        )
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")

        plt.show()

    def to_xml(
        self, filename: str, bounded_time=False, T=1.5, initial_state=None
    ) -> None:
        """Method for saving the abstraction to an XML file

        Args:
            filename (str):
                name of xml file to save to (no extension)
            bounded_time (bool, optional):
                If True, it bounds the time horizon for the
                hybrid automaton. Defaults to False.
            T (float, optional):
                Time horizon if bounded time is True . Defaults to 1.5.
            initial_state (Polyhedron, optional): adds extra mode
                corresponding to the initial state. Defaults to None.
        """
        writer = XMLWriter(self)
        writer.write(filename, bounded_time, T, initial_state)

    def to_pkl(self, filename: str):
        filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pkl(cls, filename: str) -> NeuralAbstraction:
        with open(filename, "rb") as f:
            return pickle.load(f)

    def as_smt(self, x, And):
        raise NotImplementedError

    def error_analysis(self, verbose=True) -> Stats:
        error = []
        for mode in self.modes.values():
            error.append(mode.disturbance)
        error = np.array(error, dtype=float)
        stats = Stats(
            np.mean(error, axis=0),
            np.std(error, axis=0),
            np.max(error, axis=0),
            np.min(error, axis=0),
        )
        if verbose:
            print("Error analysis: ")
            print(stats)
        return stats


class SigmoidNeuralAbstraction:
    def __init__(self, net, error, benchmark) -> None:
        self.nets = net
        self.reconstructed_net = nn.ReconstructedRelu(net)
        self.error = error
        self.ndim = benchmark.dimension
        self.benchmark = benchmark
        self.locations = [1]

    def get_symbolic(self):
        sig = verifier.DRealVerifier.sigmoid
        x = verifier.DRealVerifier.new_vars(self.ndim)
        t = translator.Translator(x, sig, sig)
        return t.translate(self.nets[0])

    def to_flowstar(self, output_file: str, config, flowstar_config):
        output_file += ".model"
        N = self.get_symbolic()
        writer = FlowstarWriter(self)
        writer.write(output_file, N, config.initial, flowstar_config)

    @staticmethod
    def get_timer():
        return T

    def plot(self, fname=None, label=False, show=True):
        domain = self.benchmark.domain
        net = self.nets[0]
        xb = [domain.lower_bounds[0], domain.upper_bounds[0]]
        yb = [domain.lower_bounds[1], domain.upper_bounds[1]]
        sysplot.plot_nn_vector_field(net, xb, yb)
        plt.xticks([-1, 0, 1])
        plt.yticks([-1, 0, 1])
        plt.xlim(xb)
        plt.ylim(yb)
        plt.gca().set_aspect("equal")
        if fname is not None:
            plt.savefig(fname + ".pdf", bbox_inches="tight")
        else:
            plt.show()

    def error_analysis(self, verbose=True) -> Stats:
        e = np.array(self.error)
        error = Stats(e, 0, e, e)
        return error


class TanhNeuralAbstraction(SigmoidNeuralAbstraction):
    def __init__(self, net, error, benchmark) -> None:
        self.nets = net
        self.error = error
        self.ndim = benchmark.dimension
        self.benchmark = benchmark
        self.locations = [1]

    @timer(T)
    def get_symbolic(self):
        tanh = verifier.DRealVerifier.tanh
        x = verifier.DRealVerifier.new_vars(self.ndim)
        t = translator.Translator(x, tanh, tanh)
        return t.translate(self.nets[0])


class NeuronTree:
    """Represents all possible activation configurations as a tree,
    and contains a depth-first search algorithm to find the valid
    ones."""

    def __init__(self, W, b, domain) -> None:
        """Initializes the tree.

        Args:
            W (List(np.ndarray)): list of weight matrices.
            b (List(np.ndarray)): list of bias vectors.
            domain (domains.Rectangle): domain of the abstraction
        """
        self.W = W
        self.b = b
        self.domain = domain
        self.layer_struct = [W[i].shape[0] for i in range(len(W) - 1)]
        self.current_path = []
        (
            self.current_ineq_constr,
            self.current_eq_constr,
        ) = construct_empty_constraint_matrices(self.W, self.b)
        self.queue = queue.Queue()
        self.configs = []
        self.stack = queue.LifoQueue()

    def check_mode(self, neuron, domain_constr):
        """Checks if a neuron is ON or OFF or both at the current location in the tree.

        Args:
            neuron (dict): neuron to check
            domain_constr (np.ndarray): current domain constraints
        """
        # Check off halfspace
        off_mode = None
        neuron_count = neuron["neuron_count"]
        self.current_path.append(0)
        a = self.current_path + [1] * (sum(self.layer_struct) - len(self.current_path))
        ineq_constr, eq_constr = get_constraints(
            self.current_ineq_constr, self.current_eq_constr, self.W, self.b, a
        )
        ineq_constr = ineq_constr[: len(self.current_path), :]
        ineq_constr = np.vstack([domain_constr, ineq_constr])
        if eq_constr is not None:
            index = min(len(self.current_path), eq_constr.shape[0])
            if index > 0:
                eq_constr = eq_constr[:index, :]
        P_off = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
        off_mode = P_off.is_nonempty()
        self.current_path.pop()

        # Check on halfpsace
        on_mode = None
        self.current_path.append(1)
        a = self.current_path + [1] * (sum(self.layer_struct) - len(self.current_path))
        ineq_constr, eq_constr = get_constraints(
            self.current_ineq_constr, self.current_eq_constr, self.W, self.b, a
        )
        ineq_constr = ineq_constr[: len(self.current_path), :]
        ineq_constr = np.vstack([domain_constr, ineq_constr])
        if eq_constr is not None:
            index = min(len(self.current_path), eq_constr.shape[0])
            if index > 0:
                eq_constr = eq_constr[:index, :]
        P_on = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
        on_mode = P_on.is_nonempty()
        self.current_path.pop()

        if off_mode and on_mode:
            neuron["mode"] = 0
            # Add neuron to stack with opposite mode to return to later
            self.stack.put(neuron)
            self.current_path.append(1)
        elif on_mode:
            self.current_path.append(1)
        else:
            assert off_mode == True
            self.current_path.append(0)

    def explore_tree(self):
        """Explores the tree using a depth first approach.

        Finds all valid activation configurations."""
        neuron_count = 0
        for hi, layer in enumerate(self.layer_struct):
            for ni in range(layer):
                neuron_count += 1
                self.queue.put(
                    {
                        "layer": hi,
                        "neuron_index": ni,
                        "mode": None,
                        "neuron_count": copy.deepcopy(neuron_count),
                    }
                )

        domain_constr = get_domain_constraints(
            self.domain,
            self.W[0].shape[1],
            nvars=(self.current_ineq_constr.shape[1] - 1),
        )
        while not (self.queue.empty()) or not self.stack.empty():
            if not self.queue.empty():
                # Still going down tree (towards leaf)
                neuron = self.queue.get()
                self.check_mode(neuron, domain_constr)
            else:
                # End of path. Save config and go back to last branching
                assert not self.stack.empty()
                assert len(self.current_path) == sum(self.layer_struct)
                self.configs.append(copy.deepcopy(self.current_path))
                neuron = self.stack.get()
                neuron_count = neuron["neuron_count"]
                self.current_path = self.current_path[: neuron_count - 1]
                self.current_path.append(0)
                for hi, layer in enumerate(self.layer_struct):
                    for ni in range(layer):
                        if hi == neuron["layer"]:
                            if ni > neuron["neuron_index"]:
                                neuron_count += 1
                                self.queue.put(
                                    {
                                        "layer": hi,
                                        "neuron_index": ni,
                                        "mode": None,
                                        "neuron_count": copy.deepcopy(neuron_count),
                                    }
                                )
                        if hi > neuron["layer"]:
                            neuron_count += 1
                            self.queue.put(
                                {
                                    "layer": hi,
                                    "neuron_index": ni,
                                    "mode": None,
                                    "neuron_count": copy.deepcopy(neuron_count),
                                }
                            )
        a = self.current_path + [1] * (sum(self.layer_struct) - len(self.current_path))
        ineq_constr, eq_constr = get_constraints(
            self.current_ineq_constr, self.current_eq_constr, self.W, self.b, a
        )
        ineq_constr = ineq_constr[: len(self.current_path), :]
        ineq_constr = np.vstack([domain_constr, ineq_constr])
        if eq_constr is not None:
            eq_constr = eq_constr[: len(self.current_path), :]
        # print(self.current_path)
        P = polyhedrons.Polyhedron(ineq_constr, L=eq_constr)
        mode = P.is_nonempty()
        if self.current_path not in self.configs:
            if mode:
                self.configs.append(copy.deepcopy(self.current_path))
        else:
            print("It happened")
        return self.configs


def all_sat(W, b, domain):
    """SMT-based algorithm for finding all active neuron configurations.

    Finds all active neuron configurations of the ReLUNet described by
    W and b in the domain.

    Args:
        W (List[np.ndarray]): list of weight matrices.
        b (List[np.ndarray]): list of bias vectors.
        domain (domains.Rectangle): domain of abstraction.

    Returns:
        configs (List): list of active neuron configurations.
    """
    ndim = W[0].shape[1]
    configs = []
    x = np.array([z3.Real("x" + str(i)) for i in range(ndim)]).reshape(-1, 1)
    Nx = x
    neurons_enabled = []
    for Wi, bi in zip(W[:-1], b[:-1]):
        Nx = Wi @ Nx + bi.reshape(-1, 1)
        neurons = [neuron > 0 for neuron in Nx.squeeze(axis=1)]
        neurons_enabled.extend(neurons)
        Nx = verifier.Z3Verifier.relu(Nx)
    Nx = W[-1] @ Nx + b[-1].reshape(-1, 1)
    XD = domain.generate_domain(x.reshape(-1).tolist(), z3.And)
    solver = z3.Solver()
    solver.add(XD)
    while solver.check() == z3.sat:
        witness = solver.model()
        config = [int(z3.is_true(witness.eval(ni))) for ni in neurons_enabled]
        configs.append(config)
        symbolic_config = [
            z3.simplify(z3.Not(ni)) if c == 0 else ni
            for ni, c in zip(neurons_enabled, config)
        ]
        solver.add(z3.Not(z3.And(symbolic_config)))
    return configs
