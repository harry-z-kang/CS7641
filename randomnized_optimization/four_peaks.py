#!/bin/env python3

import common_functions as cf

cf.plot_increasing_problem_size(
    name="Four Peaks",
    problem_size_range=range(5, 126, 20),
    max_attempts=[1000, 1000, 1000, 1000],
    max_iters=[100000, 100000, 100000, 100000],
    mimic_pop_size=500
)

cf.plot_increasing_iterations(
    name="Four Peaks",
    problem_length=100,
    max_attempts=[1000, 1000, 1000, 1000],
    max_iters=[100000, 100000, 100000, 100000],
    mimic_pop_size=500
)

cf.plot_changing_hyper_parameters(
    name="Four Peaks",
    problem_length=40,
    max_attempts=[1000, 1000, 1000, 1000],
    max_iters=[4000, 10000, 125, 40]
)
