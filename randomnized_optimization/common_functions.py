import enum
import math
import time
import mlrose_hiive

import numpy as np
import matplotlib.pyplot as plt


class ROF(enum.Enum):
    SA = 0
    RHC = 1
    GA = 2
    MIMIC = 3


def get_fitness_function(problem: str, threshold: float):
    match problem:
      case "Four Peaks":
        return mlrose_hiive.FourPeaks(t_pct=threshold)
      case "Flip Flop":
        return mlrose_hiive.FlipFlop()
      case "Continuous Peak":
        return mlrose_hiive.ContinuousPeaks(t_pct=threshold)


def plot_increasing_problem_size(
    name: str, problem_size_range: range,
    max_attempts: list[int], max_iters: list[int],
    mimic_pop_size: int
):
    fitness_list: list[list[int]] = [[], [], [], []]
    time_list: list[list[float]] = [[], [], [], []]

    for value in problem_size_range:
        print(
            f"==================== Problem Size: {value} ====================")
        fitness = get_fitness_function(name, 0.1)
        problem = mlrose_hiive.DiscreteOpt(
            length=value, fitness_fn=fitness,
            maximize=True, max_val=2
        )
        problem.set_mimic_fast_mode(True)
        init_state = np.random.randint(2, size=value)

        start = time.time()
        _, best_fitness_sa, _ = mlrose_hiive.simulated_annealing(
            problem, schedule=mlrose_hiive.ExpDecay(),
            max_attempts=max_attempts[ROF.SA.value],
            max_iters=max_iters[ROF.SA.value],
            init_state=init_state, curve=True
        )
        end = time.time()
        sa_time = end - start
        print(f"SA: {sa_time}")

        start = time.time()
        _, best_fitness_rhc, _ = mlrose_hiive.random_hill_climb(
            problem,
            max_attempts=max_attempts[ROF.RHC.value],
            max_iters=max_iters[ROF.RHC.value],
            init_state=init_state, curve=True
        )
        end = time.time()
        rhc_time = end - start
        print(f"RHC: {rhc_time}")

        start = time.time()
        _, best_fitness_ga, _ = mlrose_hiive.genetic_alg(
            problem,
            max_attempts=max_attempts[ROF.GA.value],
            max_iters=max_iters[ROF.GA.value],
            curve=True
        )
        end = time.time()
        ga_time = end - start
        print(f"GA: {ga_time}")

        start = time.time()
        _, best_fitness_mimic, _ = mlrose_hiive.mimic(
            problem, pop_size=mimic_pop_size,
            max_attempts=max_attempts[ROF.MIMIC.value],
            max_iters=max_iters[ROF.MIMIC.value],
            curve=True
        )
        end = time.time()
        mimic_time = end - start
        print(f"MIMIC: {mimic_time}")

        fitness_list[ROF.SA.value].append(best_fitness_sa)
        fitness_list[ROF.RHC.value].append(best_fitness_rhc)
        fitness_list[ROF.GA.value].append(best_fitness_ga)
        fitness_list[ROF.MIMIC.value].append(best_fitness_mimic)

        time_list[ROF.SA.value].append(sa_time)
        time_list[ROF.RHC.value].append(rhc_time)
        time_list[ROF.GA.value].append(ga_time)
        time_list[ROF.MIMIC.value].append(mimic_time)

    plt.figure()
    plt.plot(problem_size_range, fitness_list[ROF.SA.value],
             label='Simulated Annealing')
    plt.plot(problem_size_range, fitness_list[ROF.RHC.value],
             label='Randomized Hill Climb')
    plt.plot(problem_size_range, fitness_list[ROF.GA.value],
             label='Genetic Algorithm')
    plt.plot(problem_size_range, fitness_list[ROF.MIMIC.value], label='MIMIC')
    plt.title(f'Fitness vs. Problem Size ({name})')
    plt.legend()
    plt.xlabel('Problem Size')
    plt.ylabel('Fitness')
    plt.savefig(f'images/{"_".join(name.split(" ")).lower()}_fitness.png')

    plt.figure()
    plt.plot(problem_size_range, time_list[ROF.SA.value],
             label='Simulated Annealing')
    plt.plot(problem_size_range, time_list[ROF.RHC.value],
             label='Randomized Hill Climb')
    plt.plot(problem_size_range, time_list[ROF.GA.value],
             label='Genetic Algorithm')
    plt.plot(problem_size_range, time_list[ROF.MIMIC.value], label='MIMIC')
    plt.title(f'Time Efficiency vs. Problem Size ({name})')
    plt.legend()
    plt.xlabel('Problem Size')
    plt.ylabel('Computation Time (s)')
    plt.savefig(f'images/{"_".join(name.split(" ")).lower()}_computation.png')


def plot_increasing_iterations(
    name: str, problem_length: int,
    max_attempts: list[int], max_iters: list[int],
    mimic_pop_size: int
):
    fitness = get_fitness_function(name, 0.1)
    problem = mlrose_hiive.DiscreteOpt(
        length=problem_length, fitness_fn=fitness,
        maximize=True, max_val=2
    )
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=problem_length)

    _, _, fitness_curve_sa = mlrose_hiive.simulated_annealing(
        problem, schedule=mlrose_hiive.ExpDecay(),
        max_attempts=max_attempts[ROF.SA.value],
        max_iters=max_iters[ROF.SA.value],
        init_state=init_state, curve=True
    )
    print("Done with SA iterations!")
    _, _, fitness_curve_rhc = mlrose_hiive.random_hill_climb(
        problem, max_attempts=max_attempts[ROF.RHC.value],
        max_iters=max_iters[ROF.RHC.value],
        init_state=init_state, curve=True
    )
    print("Done with RHC iterations!")
    _, _, fitness_curve_ga = mlrose_hiive.genetic_alg(
        problem,
        max_attempts=max_attempts[ROF.GA.value],
        max_iters=max_iters[ROF.GA.value],
        curve=True
    )
    print("Done with GA iterations!")
    _, _, fitness_curve_mimic = mlrose_hiive.mimic(
        problem, pop_size=mimic_pop_size,
        max_attempts=max_attempts[ROF.MIMIC.value],
        max_iters=max_iters[ROF.MIMIC.value],
        curve=True
    )
    print("Done with MIMIC iterations!")

    plt.figure()
    plt.plot(fitness_curve_sa[:, 0], label='Simulated Annealing')
    plt.plot(fitness_curve_rhc[:, 0], label='Randomized Hill Climb')
    plt.plot(fitness_curve_ga[:, 0], label='Genetic Algorithm')
    plt.plot(fitness_curve_mimic[:, 0], label='MIMIC')
    plt.title(f'Fitness Curve ({name})')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig(f'images/{"_".join(name.split(" ")).lower()}_iterations.png')


def plot_changing_hyper_parameters(
    name: str, problem_length: int,
    max_attempts: list[int], max_iters: list[int]
):
    fitness = get_fitness_function(name, 0.1)
    problem = mlrose_hiive.DiscreteOpt(
        length=problem_length, fitness_fn=fitness,
        maximize=True, max_val=2
    )
    problem.set_mimic_fast_mode(True)
    init_state = np.random.randint(2, size=problem_length)

    _, _, fitness_curve_sa_1 = mlrose_hiive.simulated_annealing(
        problem, schedule=mlrose_hiive.ExpDecay(),
        max_attempts=max_attempts[ROF.SA.value],
        max_iters=max_iters[ROF.SA.value],
        init_state=init_state, curve=True
    )
    _, _, fitness_curve_sa_2 = mlrose_hiive.simulated_annealing(
        problem, schedule=mlrose_hiive.GeomDecay(),
        max_attempts=max_attempts[ROF.SA.value],
        max_iters=max_iters[ROF.SA.value],
        init_state=init_state, curve=True
    )
    _, _, fitness_curve_sa_3 = mlrose_hiive.simulated_annealing(
        problem, schedule=mlrose_hiive.ArithDecay(),
        max_attempts=max_attempts[ROF.SA.value],
        max_iters=max_iters[ROF.SA.value],
        init_state=init_state, curve=True
    )
    print("Completed SA hyper-parameter testing!")

    plt.figure()
    plt.plot(fitness_curve_sa_1[:, 0], label='decay = Exponential')
    plt.plot(fitness_curve_sa_2[:, 0], label='decay = Geometric')
    plt.plot(fitness_curve_sa_3[:, 0], label='decay = Arithmetic')
    plt.title(f'Simulated Annealing Analysis ({name})')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig(f'images/{"_".join(name.split(" ")).lower()}_sa.png')

    _, _, fitness_curve_rhc_1 = mlrose_hiive.random_hill_climb(
        problem, restarts=0,
        max_attempts=max_attempts[ROF.RHC.value],
        max_iters=max_iters[ROF.RHC.value],
        init_state=init_state, curve=True
    )
    _, _, fitness_curve_rhc_2 = mlrose_hiive.random_hill_climb(
        problem, restarts=4,
        max_attempts=max_attempts[ROF.RHC.value],
        max_iters=max_iters[ROF.RHC.value],
        init_state=init_state, curve=True
    )
    _, _, fitness_curve_rhc_3 = mlrose_hiive.random_hill_climb(
        problem, restarts=8,
        max_attempts=max_attempts[ROF.RHC.value],
        max_iters=max_iters[ROF.RHC.value],
        init_state=init_state, curve=True
    )
    _, _, fitness_curve_rhc_4 = mlrose_hiive.random_hill_climb(
        problem, restarts=12,
        max_attempts=max_attempts[ROF.RHC.value],
        max_iters=max_iters[ROF.RHC.value],
        init_state=init_state, curve=True
    )
    _, _, fitness_curve_rhc_5 = mlrose_hiive.random_hill_climb(
        problem, restarts=16,
        max_attempts=max_attempts[ROF.RHC.value],
        max_iters=max_iters[ROF.RHC.value],
        init_state=init_state, curve=True
    )
    print("Completed RHC hyper-parameter testing!")

    plt.figure()
    plt.plot(fitness_curve_rhc_1[:, 0], label='restarts = 0')
    plt.plot(fitness_curve_rhc_2[:, 0], label='restarts = 4')
    plt.plot(fitness_curve_rhc_3[:, 0], label='restarts = 8')
    plt.plot(fitness_curve_rhc_4[:, 0], label='restarts = 12')
    plt.plot(fitness_curve_rhc_5[:, 0], label='restarts = 16')
    plt.title(f'Randomized Hill Climb Analysis ({name})')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig(f'images/{"_".join(name.split(" ")).lower()}_rhc.png')

    _, _, fitness_curve_mimic_1 = mlrose_hiive.mimic(
        problem, keep_pct=0.1,
        pop_size=100,
        max_attempts=max_attempts[ROF.MIMIC.value],
        max_iters=max_iters[ROF.MIMIC.value],
        curve=True
    )
    _, _, fitness_curve_mimic_2 = mlrose_hiive.mimic(
        problem, keep_pct=0.3,
        pop_size=100,
        max_attempts=max_attempts[ROF.MIMIC.value],
        max_iters=max_iters[ROF.MIMIC.value],
        curve=True
    )
    _, _, fitness_curve_mimic_3 = mlrose_hiive.mimic(
        problem, keep_pct=0.1,
        pop_size=200,
        max_attempts=max_attempts[ROF.MIMIC.value],
        max_iters=max_iters[ROF.MIMIC.value],
        curve=True
    )
    _, _, fitness_curve_mimic_4 = mlrose_hiive.mimic(
        problem, keep_pct=0.3,
        pop_size=200,
        max_attempts=max_attempts[ROF.MIMIC.value],
        max_iters=max_iters[ROF.MIMIC.value],
        curve=True
    )
    _, _, fitness_curve_mimic_5 = mlrose_hiive.mimic(
        problem, keep_pct=0.1,
        pop_size=500,
        max_attempts=max_attempts[ROF.MIMIC.value],
        max_iters=max_iters[ROF.MIMIC.value],
        curve=True
    )
    _, _, fitness_curve_mimic_6 = mlrose_hiive.mimic(
        problem, keep_pct=0.3,
        pop_size=500,
        max_attempts=max_attempts[ROF.MIMIC.value],
        max_iters=max_iters[ROF.MIMIC.value],
        curve=True
    )
    print("Completed MIMIC hyper-parameter testing!")

    plt.figure()
    plt.plot(fitness_curve_mimic_1[:, 0],
             label='keep % = 0.1, population = 100')
    plt.plot(fitness_curve_mimic_2[:, 0],
             label='keep % = 0.3, population = 100')
    plt.plot(fitness_curve_mimic_3[:, 0],
             label='keep % = 0.1, population = 200')
    plt.plot(fitness_curve_mimic_4[:, 0],
             label='keep % = 0.3, population = 200')
    plt.plot(fitness_curve_mimic_5[:, 0],
             label='keep % = 0.1, population = 500')
    plt.plot(fitness_curve_mimic_6[:, 0],
             label='keep % = 0.3, population = 500')
    plt.title(f'MIMIC Analysis ({name})')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig(f'images/{"_".join(name.split(" ")).lower()}_mimic.png')

    _, _, fitness_curve_ga_1 = mlrose_hiive.genetic_alg(
        problem, mutation_prob=0.1, pop_size=100,
        max_attempts=max_attempts[ROF.GA.value],
        max_iters=max_iters[ROF.GA.value],
        curve=True
    )
    _, _, fitness_curve_ga_2 = mlrose_hiive.genetic_alg(
        problem, mutation_prob=0.3, pop_size=100,
        max_attempts=max_attempts[ROF.GA.value],
        max_iters=max_iters[ROF.GA.value],
        curve=True
    )
    _, _, fitness_curve_ga_3 = mlrose_hiive.genetic_alg(
        problem, mutation_prob=0.1, pop_size=200,
        max_attempts=max_attempts[ROF.GA.value],
        max_iters=max_iters[ROF.GA.value],
        curve=True
    )
    _, _, fitness_curve_ga_4 = mlrose_hiive.genetic_alg(
        problem, mutation_prob=0.3, pop_size=200,
        max_attempts=max_attempts[ROF.GA.value],
        max_iters=max_iters[ROF.GA.value],
        curve=True
    )
    _, _, fitness_curve_ga_5 = mlrose_hiive.genetic_alg(
        problem, mutation_prob=0.1, pop_size=500,
        max_attempts=max_attempts[ROF.GA.value],
        max_iters=max_iters[ROF.GA.value],
        curve=True
    )
    _, _, fitness_curve_ga_6 = mlrose_hiive.genetic_alg(
        problem, mutation_prob=0.3, pop_size=500,
        max_attempts=max_attempts[ROF.GA.value],
        max_iters=max_iters[ROF.GA.value],
        curve=True
    )
    print("Completed GA hyper-parameter testing!")

    plt.figure()
    plt.plot(fitness_curve_ga_1[:, 0],
             label='mutation prob = 0.1, population = 100')
    plt.plot(fitness_curve_ga_2[:, 0],
             label='mutation prob = 0.3, population = 100')
    plt.plot(fitness_curve_ga_3[:, 0],
             label='mutation prob = 0.1, population = 200')
    plt.plot(fitness_curve_ga_4[:, 0],
             label='mutation prob = 0.3, population = 200')
    plt.plot(fitness_curve_ga_5[:, 0],
             label='mutation prob = 0.1, population = 500')
    plt.plot(fitness_curve_ga_6[:, 0],
             label='mutation prob = 0.3, population = 500')
    plt.title(f'GA Analysis ({name})')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.savefig(f'images/{"_".join(name.split(" ")).lower()}_ga.png')
