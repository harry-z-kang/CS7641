#!/bin/env python3

import numpy as np
import common_functions as cf

cf.plot_increasing_problem_size(
    name="Flip Flop",
    problem_size_range=range(5, 125, 20),
    max_attempts=[10, 10, 10, 10],
    max_iters=[2000, 2000, np.inf, np.inf],
    mimic_pop_size=300
)

cf.plot_increasing_iterations(
    name="Flip Flop",
    problem_length=100,
    max_attempts=[10, 10, 10, 10],
    max_iters=[2000, 2000, np.inf, np.inf],
    mimic_pop_size=300
)

cf.plot_different_thresholds(
    name="Flip Flop",
    problem_length=40,
    threshold_range=[0.1, 0.3, 0.5],
    max_attempts=[10, 10, 10, 10],
    max_iters=[2000, 2000, np.inf, np.inf],
    mimic_pop_size=300
)

cf.plot_changing_hyper_parameters(
    name="Flip Flop",
    problem_length=40,
    max_attempts=[10, 10, 10, 10],
    max_iters=[2000, 2000, np.inf, np.inf]
)
