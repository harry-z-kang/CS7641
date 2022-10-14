#!/bin/env python3

import enum
import time
import mlrose_hiive
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dataclasses import dataclass
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/breast-cancer-wisconsin.data')
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

features = preprocessing.scale(data.drop(['diagnosis'], axis=1))
target = data["diagnosis"]

print(f"Number of samples: {target.size}")
print(
    f"Percentage of Malignment: {target.value_counts(normalize=True)['M'] * 100:.3f}%")
print(
    f"Percentage of Benign: {target.value_counts(normalize=True)['B'] * 100:.3f}%")

data["diagnosis"] = preprocessing.OrdinalEncoder(
).fit_transform(data[["diagnosis"]])
target = data["diagnosis"].astype(int)

print("Splitting into train/validation/test sets...", end="")
FEATURES_TRAIN, FEATURES_TEST, TARGET_TRAIN, TARGET_TEST = train_test_split(
    features, target, test_size=0.2, random_state=42)
FEATURES_TRAIN, FEATURES_VAL, TARGET_TRAIN, TARGET_VAL = train_test_split(
    features, target, test_size=0.2, random_state=42)
print("Done")


@dataclass
class Stats:
    training_time: list[list[float]]
    train_acc: list[list[float]]
    val_acc: list[list[float]]
    test_acc: list[list[float]]
    best_model_pred: np.ndarray
    best_model_fitness: np.ndarray
    best_model_index: tuple[int, int]

    @property
    def train_acc_best(self) -> float:
        return self.train_acc[self.best_model_index[0]][self.best_model_index[1]]

    @property
    def val_acc_best(self) -> float:
        return self.val_acc[self.best_model_index[0]][self.best_model_index[1]]

    @property
    def test_acc_best(self) -> float:
        return self.test_acc[self.best_model_index[0]][self.best_model_index[1]]

    @property
    def time_best(self) -> float:
        return self.training_time[self.best_model_index[0]][self.best_model_index[1]]


class OptimizationAlgo(enum.Enum):
    RHC = "random_hill_climb"
    SA = "simulated_annealing"
    GA = "genetic_alg"
    BP = "gradient_descent"


TARGET_CLASSES = ['B', 'M']


def plot_confusion_matrix(predicted, expected, algo_name) -> None:
    print(f'{algo_name} Confusion matrix')
    cf = confusion_matrix(predicted, expected)
    sns.heatmap(cf, annot=True, yticklabels=TARGET_CLASSES,
                xticklabels=TARGET_CLASSES, cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.savefig(f"images/confusion_matrix_{algo_name}_best.png")


LEARNING_RATE = [0.00001, 0.0001, 0.01, 0.1, 1]


def run_optimization(algorithm: OptimizationAlgo, stat: Stats, tuned_param_values: list = [0], tuned_param_name: str = ""):
    for i, learning_rate in enumerate(LEARNING_RATE):
        for j, param in enumerate(tuned_param_values):
            match algorithm:
                case OptimizationAlgo.RHC:
                    nn_model = mlrose_hiive.NeuralNetwork(
                        algorithm=algorithm.value,
                        hidden_nodes=[16], activation='relu',
                        max_iters=2000, bias=True, is_classifier=True,
                        learning_rate=learning_rate, early_stopping=True,
                        max_attempts=100, random_state=42, curve=True, restarts=param
                    )
                case OptimizationAlgo.SA:
                    nn_model = mlrose_hiive.NeuralNetwork(
                        algorithm=algorithm.value,
                        hidden_nodes=[16], activation='relu',
                        max_iters=2000, bias=True, is_classifier=True,
                        learning_rate=learning_rate, early_stopping=True,
                        max_attempts=100, random_state=42, curve=True, schedule=param
                    )
                case OptimizationAlgo.GA:
                    nn_model = mlrose_hiive.NeuralNetwork(
                        algorithm=algorithm.value,
                        hidden_nodes=[16], activation='relu',
                        max_iters=500, bias=True, is_classifier=True,
                        learning_rate=learning_rate, early_stopping=True,
                        max_attempts=100, random_state=42, curve=True, pop_size=param
                    )
                case OptimizationAlgo.BP:
                    nn_model = mlrose_hiive.NeuralNetwork(
                        algorithm=algorithm.value,
                        hidden_nodes=[16], activation='relu',
                        max_iters=1000, bias=True, is_classifier=True,
                        learning_rate=learning_rate, early_stopping=True,
                        max_attempts=100, random_state=42, curve=True
                    )

            start = time.time()
            nn_model.fit(FEATURES_TRAIN, TARGET_TRAIN)
            end = time.time()
            stat.training_time[i][j] = end - start

            target_train_pred = nn_model.predict(FEATURES_TRAIN)
            target_train_accuracy = accuracy_score(
                TARGET_TRAIN, target_train_pred)
            stat.train_acc[i][j] = target_train_accuracy

            target_val_pred = nn_model.predict(FEATURES_VAL)
            target_val_accuracy = accuracy_score(
                TARGET_VAL, target_val_pred)
            stat.val_acc[i][j] = target_val_accuracy

            target_test_pred = nn_model.predict(FEATURES_TEST)
            target_test_accuracy = accuracy_score(
                TARGET_TEST, target_test_pred)
            stat.test_acc[i][j] = target_test_accuracy

            if target_val_accuracy > stat.val_acc_best:
                print(f"Learning Rate: {learning_rate}")
                if tuned_param_name != "" and tuned_param_values != [0]:
                    print(f"{tuned_param_name}: {param}")
                print(f"Time: {stat.training_time[i][j]}")
                stat.best_model_pred = target_test_pred
                stat.best_model_fitness = nn_model.fitness_curve
                stat.best_model_index = (i, j)

            print("Iteration done!")

    plt.figure()
    try:
      plt.plot(stat.best_model_fitness[:, 0])
    except IndexError:
      plt.plot(-stat.best_model_fitness)
    plt.grid()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss vs. No. of Iterations for {algorithm.name} (Best Model)')
    plt.savefig(f'images/nn_train_iterations_curve_{algorithm.name.lower()}_best.png')

    print(f"Average Time: {np.mean(stat.training_time)}")
    print(f"Time: {stat.training_time}")
    print(f"Test Accuracy: {stat.test_acc[stat.best_model_index[0]][stat.best_model_index[1]]}")

    plot_confusion_matrix(stat.best_model_pred, TARGET_TEST, algorithm.name)

    print(f'{algorithm.name} Completed!')


restarts: list[int] = [2, 4, 6, 8, 10]
rhc_stats: Stats = Stats(
    np.zeros((len(LEARNING_RATE), len(restarts)), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), len(restarts)), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), len(restarts)), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), len(restarts)), dtype=np.float64),
    np.array(0), np.array(0), (0, 0)
)
run_optimization(OptimizationAlgo.RHC, rhc_stats, restarts, 'Restarts')

schedules = [mlrose_hiive.GeomDecay(), mlrose_hiive.ExpDecay(), mlrose_hiive.ArithDecay()]
sa_stats: Stats = Stats(
    np.zeros((len(LEARNING_RATE), len(schedules)), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), len(schedules)), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), len(schedules)), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), len(schedules)), dtype=np.float64),
    np.array(0), np.array(0), (0, 0)
)
run_optimization(OptimizationAlgo.SA, sa_stats, schedules, 'Schedule')

populations: list[int] = [20, 50, 100, 200, 500]
ga_stats: Stats = Stats(
    np.zeros((len(LEARNING_RATE), len(populations)), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), len(populations)), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), len(populations)), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), len(populations)), dtype=np.float64),
    np.array(0), np.array(0), (0, 0)
)
run_optimization(OptimizationAlgo.GA, ga_stats, populations, 'Population')

bp_stats: Stats = Stats(
    np.zeros((len(LEARNING_RATE), 1), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), 1), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), 1), dtype=np.float64),
    np.zeros((len(LEARNING_RATE), 1), dtype=np.float64),
    np.array(0), np.array(0), (0, 0)
)
run_optimization(OptimizationAlgo.BP, bp_stats)

plt.figure()
plt.bar(OptimizationAlgo.__members__.keys(), [
    rhc_stats.time_best,
    sa_stats.time_best,
    ga_stats.time_best,
    bp_stats.time_best
])
plt.xlabel("Algorithm")
plt.ylabel("Best Time (s)")
plt.title('Best Times for Algorithms')
plt.savefig('images/nn_best_times.png')

plt.figure()
plt.bar(OptimizationAlgo.__members__.keys(), [
    rhc_stats.test_acc_best,
    sa_stats.test_acc_best,
    ga_stats.test_acc_best,
    bp_stats.test_acc_best
])
plt.xlabel("Algorithm")
plt.ylabel("Best Score (%)")
plt.title('Best Test Score for Algorithms')
plt.ylim((0.9, 1.0))
plt.savefig('images/nn_best_test_scores.png')

plt.figure()
plt.bar(OptimizationAlgo.__members__.keys(), [
    rhc_stats.train_acc_best,
    sa_stats.train_acc_best,
    ga_stats.train_acc_best,
    bp_stats.train_acc_best
])
plt.xlabel("Algorithm")
plt.ylabel("Best Score (%)")
plt.title('Best Train Score for Algorithms')
plt.ylim((0.9, 1.0))
plt.savefig('images/nn_best_train_scores.png')

plt.figure()
plt.bar(OptimizationAlgo.__members__.keys(), [
    rhc_stats.val_acc_best,
    sa_stats.val_acc_best,
    ga_stats.val_acc_best,
    bp_stats.val_acc_best
])
plt.xlabel("Algorithm")
plt.ylabel("Best Score (%)")
plt.title('Best Validation Score for Algorithms')
plt.ylim((0.9, 1.0))
plt.savefig('images/nn_best_val_scores.png')
