# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Callable, List, Tuple

import numpy as np
from torch.nn import Sequential

from cegis.nn import ReluNet


class Translator:
    """Translates a torch neural network representing a dynamical model
    to a symbolic expression.
    """

    def __init__(
        self, vars: Tuple, sigma_hidden: Callable, sigma_final: Callable, rounding=5
    ) -> None:
        """Initialises the translator.

        Args:
            vars (Tuple): Symbolic variables representing the input.
            sigma_hidden (Callable): Hidden activation function for symbolic inputs
            sigma_final (Callable): Final activation function for symbolic inputs
        """
        self.input_vars = np.array(vars).reshape(-1, 1)
        self.relu = sigma_hidden  # Hidden layer relu
        self.recu = sigma_final  # Output relu (template)
        self.rounding = rounding

    def translate(self, net: ReluNet) -> np.ndarray:
        """Performs the translation of the neural network.

        Args:
            net (ReluNet): net to translate

        Returns:
            np.ndarray: vector representing the translated neural network.
        """
        dp = self.rounding
        # W_in = net.model[0].weight.detach().numpy().round(dp)
        # b_in = net.model[0].bias.detach().numpy().reshape(-1, 1).round(dp)
        W_out = net.model[-1].weight.detach().numpy().round(dp)
        b_out = net.model[-1].bias.detach().numpy().reshape(-1, 1).round(dp)

        x = self.input_vars
        # x = self.relu(x)
        for i in range(int((len(net.model) - 1) / 2) - 1):
            W = net.model[2 * i].weight.detach().numpy().round(dp)
            b = net.model[2 * i].bias.detach().numpy().reshape(-1, 1).round(dp)
            x = W @ x + b
            x = self.relu(x)
        W = net.model[-3].weight.detach().numpy().round(dp)
        b = net.model[-3].bias.detach().numpy().reshape(-1, 1).round(dp)
        x = W @ x + b
        x = self.recu(x)
        x = W_out @ x + b_out
        return x
