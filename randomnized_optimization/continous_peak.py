#!/bin/env python3

import common_functions as cf

cf.plot_increasing_problem_size(
    name="Continuous Peak",
    problem_size_range=range(5, 126, 20),
    max_attempts=[1000, 1000, 1000, 1000],
    max_iters=[100000, 100000, 100000, 100000],
    mimic_pop_size=500
)

cf.plot_increasing_iterations(
    name="Continuous Peak",
    problem_length=85,
    max_attempts=[1000, 1000, 1000, 1000],
    max_iters=[3000, 3000, 3000, 3000],
    mimic_pop_size=500
)

cf.plot_changing_hyper_parameters(
    name="Continuous Peak",
    problem_length=85,
    max_attempts=[1000, 1000, 1000, 1000],
    max_iters=[5000, 1500, 1000, 50]
)
