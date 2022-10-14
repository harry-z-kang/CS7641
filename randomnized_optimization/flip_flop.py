#!/bin/env python3

import numpy as np
import common_functions as cf

cf.plot_increasing_problem_size(
    name="Flip Flop",
    problem_size_range=range(5, 126, 20),
    max_attempts=[100, 100, 100, 100],
    max_iters=[np.inf, np.inf, np.inf, np.inf],
    mimic_pop_size=300
)

cf.plot_increasing_iterations(
    name="Flip Flop",
    problem_length=100,
    max_attempts=[100, 100, 100, 100],
    max_iters=[np.inf, np.inf, np.inf, np.inf],
    mimic_pop_size=300
)

cf.plot_changing_hyper_parameters(
    name="Flip Flop",
    problem_length=100,
    max_attempts=[100, 100, 100, 100],
    max_iters=[np.inf, np.inf, np.inf, np.inf]
)
