#!/bin/env python3

import common_functions as cf

cf.plot_increasing_problem_size(
    name="Four Peaks",
    problem_size_range=range(5, 125, 20),
    max_attempts=[1000, 1000, 1000, 100],
    max_iters=[100000, 10000, 10000, 10000],
    mimic_pop_size=500
)

cf.plot_increasing_iterations(
    name="Four Peaks",
    problem_length=100,
    max_attempts=[1000, 1000, 1000, 100],
    max_iters=[100000, 10000, 10000, 10000],
    mimic_pop_size=500
)

cf.plot_different_thresholds(
    name="Four Peaks",
    problem_length=40,
    threshold_range=[0.1, 0.3, 0.5],
    max_attempts=[1000, 1000, 1000, 100],
    max_iters=[100000, 10000, 10000, 10000],
    mimic_pop_size=500
)

cf.plot_changing_hyper_parameters(
    name="Four Peaks",
    problem_length=40,
    max_attempts=[1000, 1000, 1000, 100],
    max_iters=[100000, 10000, 10000, 10000]
)
