import unittest

import numpy as np
from matplotlib import pyplot as plt
import z3

import neural_abstraction as na
import polyhedrons as ph
import cegis.verifier as verifier


def plot_mode(M, ax=None, color="blue"):
    """Plot the mode's domain and flow"""
    P = M.P
    flow = M.flow_no_disturbance
    ax = ax or plt.gca()
    P.plot(color=color)
    data = np.array(P.box_sample(100))
    f = flow(data).T
    ax.quiver(
        data[:, 0],
        data[:, 1],
        f[:, 0],
        f[:, 1],
        # linewidth=0.8,
        # headwidth=0.5,
    )
    return ax
    # plt.show()
    # color=color)
    # cmap=sns.color_palette(
    #     "ch:s=0,rot=-.4,l=0.8,d=0",
    #     as_cmap=True,
    # ),


# P0 = ph.vertices2polyhedron([[0.1, 0.1], [1, 0.1], [1, 1], [0.1, 1]])
# P1 = ph.vertices2polyhedron([[0.1, 0.1], [-1, 0.1], [-1, -1], [0.1, -1]])
# P2 = ph.vertices2polyhedron([[0.1, 0.1], [-1, 0.1], [-1, 1], [0.1, 1]])
# P4 = ph.vertices2polyhedron([[0, 0], [1, 0], [1, 1], [0, 1]])

P0 = ph.vertices2polyhedron([[0, 0], [4, 0], [4, 3]])
P1 = ph.vertices2polyhedron([[4, 0], [5, 0], [6, 3], [4, 3]])
P2 = ph.vertices2polyhedron([[4, 3], [6, 3], [8, 5]])

f0 = np.array([[0, 0], [0, 0]]), np.array([0.1, 1])
f1 = np.array([[0, 0], [0, 0]]), np.array([-0.1, -1])
f2 = np.array([[0, 0], [0, 0]]), np.array([0.1, 1])
# A4, c4 = np.array([[1, 0], [0, 1]]), np.array([0, 0])

M0 = na.Mode(f0, P0)
M1 = na.Mode(f1, P1)
M2 = na.Mode(f2, P2, disturbance=[0.09, 0.09])

modes = {"0": M0, "1": M1, "2": M2}

transitions = [("1", "2"), ("2", "0"), ("0", "2"), ("2", "1"), ("1", "0"), ("0", "1")]

transistion_pruned = [("0", "1"), ("0", "2"), ("1", "0")]


def prune_transitions_new(transitions, modes, dim=2):
    """Prune transitions that are never taken based on flow directions"""

    x = verifier.Z3Verifier.new_vars(dim)
    new_transitions = []
    solver = z3.Solver()
    for transition in transitions:
        source_id, destination_id = transition
        source = modes[source_id]
        destination = modes[destination_id]
        intersection = z3.And(
            source.P.as_smt(x, z3.And),
            destination.P.as_smt(x, z3.And),
        )
        source_flow_upper = source.flow_upper(np.array(x).reshape(1, -1))
        source_flow_lower = source.flow_lower(np.array(x).reshape(1, -1))
        lies_in = []
        for a, b in destination.P.hyperplanes():
            on_hyperplane = np.dot(a.reshape(1, -1), x).item() == b
            lie_upper = np.dot(a.reshape(1, -1), source_flow_upper).item()
            lie_lower = np.dot(a.reshape(1, -1), source_flow_lower).item()
            f1 = z3.And(lie_lower <= 0, on_hyperplane, intersection)
            f2 = z3.And(lie_upper <= 0, on_hyperplane, intersection)
            lies_in.append(z3.Or(f1, f2))

        solver.add(z3.Or(lies_in))
        res = solver.check()
        if res == z3.sat:
            m = solver.model()
        if res == z3.unsat:
            print("Transition pruned:", transition)
        else:
            new_transitions.append(transition)
        solver.reset()

    return new_transitions


def prune_transitions(transitions, modes, dim=2):
    """Prune transitions that are never taken based on flow directions"""

    x = verifier.Z3Verifier.new_vars(dim)
    new_transitions = []
    solver = z3.Solver()
    for transition in transitions:
        source_id, destination_id = transition
        source = modes[source_id]
        destination = modes[destination_id]
        solver.add(
            source.P.as_smt(x, z3.And),
            destination.P.as_smt(x, z3.And),
        )
        solver.check()
        model = solver.model()
        source_flow = source.flow_no_disturbance(np.array(x).reshape(1, -1))
        x0 = np.array([model[x[i]].as_fraction() for i in range(dim)])
        planes = []
        shifts = []
        for a, b in source.P.hyperplanes():
            if np.dot(a, x0) - b == 0:
                planes.append(-a.reshape(-1, 1))
                shifts.append(b)
        assert len(planes) > 0
        # get point inside polyhedron
        v1 = np.array(source.P.V[0])[1:]
        v2 = np.array(source.P.V[1])[1:]
        p = (v1 + v2) / 2

        lies = [(plane.T @ (source_flow))[0, 0] for plane, b in zip(planes, shifts)]
        F = [lie <= 0 for lie in lies]
        solver.add(z3.Or(F))
        res = solver.check()
        if res == z3.unsat:
            print("Transition pruned:", transition)
        else:
            new_transitions.append(transition)

        solver.reset()

    return new_transitions


if __name__ == "__main__":
    fig, ax = plt.subplots()
    for M in [M0, M1, M2]:
        ax = plot_mode(M, ax=ax)
    plt.show()
    new_transitions = prune_transitions_new(transitions, modes)
    print(new_transitions)
