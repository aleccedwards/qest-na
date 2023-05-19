# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest

from matplotlib import pyplot as plt
import numpy as np
import torch

import neural_abstraction as na
import reach
import polyhedrons


class TestReachReg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.na = na.NeuralAbstraction.from_pkl("results/jet.pkl")

    def test_integrator(self):
        np.random.seed(1)
        torch.manual_seed(1)
        V = [[0.4, 0], [0.45, 0], [0.4, 0.03], [0.45, 0.03]]
        plt.ion()
        XI = polyhedrons.convert2template(
            polyhedrons.vertices2polyhedron(V), polyhedrons.get_2d_template(4)
        )
        mode_id = self.na.find_mode(V[0])
        mode = self.na.modes[mode_id]
        lt, ut, L = reach.estimate_time_bounds(self.na, mode, XI, delta=0.02)
        I = reach.ScipyIntegrator(mode.flow_random)
        print(lt, ut)
        S = XI.box_sample(100)
        for s in S:
            res = I.integrate(ut, s)
            Y = res.y
            plt.plot(Y[0, :], Y[1, :])
        XI.plot()
        self.na.plot()
        plt.ioff()
        plt.show()
        # plt.pause(5)

    def test_lgg(self):
        np.random.seed(1)
        torch.manual_seed(1)
        V = [[0.4, 0], [0.45, 0], [0.4, 0.03], [0.45, 0.03]]
        XI = polyhedrons.convert2template(
            polyhedrons.vertices2polyhedron(V), polyhedrons.get_2d_template(4)
        )
        XI.plot("black")
        mode_id = self.na.find_mode([0.41, 0.01])
        RX = reach.nn_reach(self.na, XI, self.na.modes[mode_id], 0.11)
        for P in RX:
            try:
                P.plot()
            except IndexError as e:
                print(e)
        plt.show()


if __name__ == "__main__":
    unittest.main()
