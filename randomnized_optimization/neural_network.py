#!/bin/env python3

import time
import mlrose_hiive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

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

learning_rates = [0.00001, 0.0001, 0.01, 0.1, 1]
restarts = [2, 4, 6, 8, 10]
schedules = [mlrose_hiive.GeomDecay(), mlrose_hiive.ExpDecay(),
             mlrose_hiive.ArithDecay()]
populations = [20, 50, 100, 200, 500]

train_acc_rhc = np.zeros((len(learning_rates), len(restarts)))
val_acc_rhc = np.zeros((len(learning_rates), len(restarts)))
val_acc_rhc_best = 0.0
rhc_best_idx1 = 0
rhc_best_idx2 = 0
test_acc_rhc = np.zeros((len(learning_rates), len(restarts)))
time_rhc = np.zeros((len(learning_rates), len(restarts)))

train_acc_sa = np.zeros((len(learning_rates), len(schedules)))
val_acc_sa = np.zeros((len(learning_rates), len(schedules)))
val_acc_sa_best = 0.0
sa_best_idx1 = 0
sa_best_idx2 = 0
test_acc_sa = np.zeros((len(learning_rates), len(schedules)))
time_sa = np.zeros((len(learning_rates), len(schedules)))

train_acc_ga = np.zeros((len(learning_rates), len(populations)))
val_acc_ga = np.zeros((len(learning_rates), len(populations)))
val_acc_ga_best = 0.0
ga_best_idx1 = 0
ga_best_idx2 = 0
test_acc_ga = np.zeros((len(learning_rates), len(populations)))
time_ga = np.zeros((len(learning_rates), len(populations)))

train_acc_backprop = np.zeros((len(learning_rates), 1))
val_acc_backprop = np.zeros((len(learning_rates), 1))
val_acc_backprop_best = 0.0
backprop_best_idx1 = 0
backprop_best_idx2 = 0
test_acc_backprop = np.zeros((len(learning_rates), 1))
time_backprop = np.zeros((len(learning_rates), 1))

# RHC
nn_model_rhc_best = mlrose_hiive.NeuralNetwork(
    hidden_nodes=[16], activation='relu',
    algorithm='random_hill_climb',
    max_iters=2000, bias=True, is_classifier=True,
    learning_rate=0.00001, early_stopping=True,
    max_attempts=100, random_state=42, curve=True
)

for i, learning_rate in enumerate(learning_rates):
    for j, restart in enumerate(restarts):
        nn_model_rhc = mlrose_hiive.NeuralNetwork(
            hidden_nodes=[16], activation='relu',
            algorithm='random_hill_climb',
            max_iters=2000, bias=True, restarts=restart, is_classifier=True,
            learning_rate=learning_rate, early_stopping=True,
            max_attempts=100, random_state=42, curve=True
        )

        start = time.time()
        nn_model_rhc.fit(FEATURES_TRAIN, TARGET_TRAIN)
        end = time.time()
        time_rhc_current = end - start
        time_rhc[i][j] = time_rhc_current

        target_train_pred_rhc = nn_model_rhc.predict(FEATURES_TRAIN)
        target_train_accuracy_rhc = accuracy_score(
            TARGET_TRAIN, target_train_pred_rhc)
        train_acc_rhc[i][j] = target_train_accuracy_rhc

        target_val_pred_rhc = nn_model_rhc.predict(FEATURES_VAL)
        target_val_accuracy_rhc = accuracy_score(
            TARGET_VAL, target_val_pred_rhc)
        val_acc_rhc[i][j] = target_val_accuracy_rhc

        target_test_pred_rhc = nn_model_rhc.predict(FEATURES_TEST)
        target_test_accuracy_rhc = accuracy_score(
            TARGET_TEST, target_test_pred_rhc)
        test_acc_rhc[i][j] = target_test_accuracy_rhc

        if target_val_accuracy_rhc > val_acc_rhc_best:
            nn_model_rhc_best = nn_model_rhc
            print("Learning Rate:", learning_rate)
            print("Restarts:", restart)
            print("Time:", time_rhc_current)
            rhc_best_idx1 = i
            rhc_best_idx2 = j
            val_acc_rhc_best = target_val_accuracy_rhc

        print("Iteration done!")

plt.figure()
plt.plot(nn_model_rhc_best.fitness_curve[:, 0])
plt.grid()
plt.xlabel('Number of Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. No. of Iterations for RHC (Best Model)')
plt.savefig('images/nn_train_iterations_curve_rhc_best.png')

target_test_pred_rhc = nn_model_rhc_best.predict(FEATURES_TEST)
confusion_matrix_rhc = confusion_matrix(TARGET_TEST, target_test_pred_rhc)

print("Average Time", np.mean(time_rhc))
print("Time:", time_rhc)
print("Test Accuracy:", test_acc_rhc[rhc_best_idx1][rhc_best_idx2])
print("Confusion Matrix:", confusion_matrix_rhc)

print('RHC Completed!')

# SA
nn_model_sa_best = mlrose_hiive.NeuralNetwork(
    hidden_nodes=[16], activation='relu',
    algorithm='simulated_annealing',
    max_iters=2000, bias=True, is_classifier=True,
    learning_rate=0.00001, early_stopping=True,
    max_attempts=100, random_state=42, curve=True
)

for i, learning_rate in enumerate(learning_rates):
    for j, schedule in enumerate(schedules):
        nn_model_sa = mlrose_hiive.NeuralNetwork(
            hidden_nodes=[16], activation='relu',
            algorithm='simulated_annealing',
            max_iters=2000, bias=True, schedule=schedule, is_classifier=True,
            learning_rate=learning_rate, early_stopping=True,
            max_attempts=100, random_state=42, curve=True
        )

        start = time.time()
        nn_model_sa.fit(FEATURES_TRAIN, TARGET_TRAIN)
        end = time.time()
        time_sa_current = end - start

        target_train_pred_sa = nn_model_sa.predict(FEATURES_TRAIN)
        target_train_accuracy_sa = accuracy_score(
            TARGET_TRAIN, target_train_pred_sa)
        train_acc_sa[i][j] = target_train_accuracy_sa

        target_val_pred_sa = nn_model_sa.predict(FEATURES_VAL)
        target_val_accuracy_sa = accuracy_score(
            TARGET_VAL, target_val_pred_sa)
        val_acc_sa[i][j] = target_val_accuracy_sa

        target_test_pred_sa = nn_model_sa.predict(FEATURES_TEST)
        target_test_accuracy_sa = accuracy_score(
            TARGET_TEST, target_test_pred_sa)
        test_acc_sa[i][j] = target_test_accuracy_sa
        time_sa[i][j] = time_sa_current

        if target_val_accuracy_sa > val_acc_sa_best:
            nn_model_sa_best = nn_model_sa
            print("Learning Rate:", learning_rate)
            print("Schedule:", j)
            print("Time:", time_sa_current)
            sa_best_idx1 = i
            sa_best_idx2 = j
            val_acc_sa_best = target_val_accuracy_sa

        print("Iteration done!")

plt.figure()
plt.plot(nn_model_sa_best.fitness_curve[:, 0])
plt.grid()
plt.xlabel('Number of Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. No. of Iterations for SA (Best Model)')
plt.savefig('images/nn_train_iterations_curve_sa_best.png')

target_test_pred_sa = nn_model_sa_best.predict(FEATURES_TEST)
confusion_matrix_sa = confusion_matrix(TARGET_TEST, target_test_pred_sa)

print("Average Time", np.mean(time_sa))
print("Time:", time_sa)
print("Test Accuracy:", test_acc_sa[sa_best_idx1][sa_best_idx2])
print("Confusion Matrix:", confusion_matrix_sa)

print('SA Completed!')

# Back-Propogation
nn_model_backprop_best = mlrose_hiive.NeuralNetwork(
    hidden_nodes=[16], activation='relu',
    algorithm='gradient_descent',
    max_iters=1000, bias=True, is_classifier=True,
    learning_rate=0.00001, early_stopping=True,
    max_attempts=100, random_state=42, curve=True
)

for i, learning_rate in enumerate(learning_rates):
    for j in range(1):
        nn_model_backprop = mlrose_hiive.NeuralNetwork(
            hidden_nodes=[16], activation='relu',
            algorithm='gradient_descent',
            max_iters=1000, bias=True, is_classifier=True,
            learning_rate=learning_rate, early_stopping=True,
            max_attempts=100, random_state=42, curve=True
        )

        start = time.time()
        nn_model_backprop.fit(FEATURES_TRAIN, TARGET_TRAIN)
        end = time.time()
        time_backprop_current = end - start

        target_train_pred_backprop = nn_model_backprop.predict(FEATURES_TRAIN)
        target_train_accuracy_backprop = accuracy_score(
            TARGET_TRAIN, target_train_pred_backprop)
        train_acc_backprop[i][j] = target_train_accuracy_backprop

        target_val_pred_backprop = nn_model_backprop.predict(FEATURES_VAL)
        target_val_accuracy_backprop = accuracy_score(
            TARGET_VAL, target_val_pred_backprop)
        val_acc_backprop[i][j] = target_val_accuracy_backprop

        target_test_pred_backprop = nn_model_backprop.predict(FEATURES_TEST)
        target_test_accuracy_backprop = accuracy_score(
            TARGET_TEST, target_test_pred_backprop)
        test_acc_backprop[i][j] = target_test_accuracy_backprop
        time_backprop[i][j] = time_backprop_current

        if target_val_accuracy_backprop > val_acc_backprop_best:
            nn_model_backprop_best = nn_model_backprop
            print("Learning Rate:", learning_rate)
            print("Time:", time_backprop_current)
            backprop_best_idx1 = i
            backprop_best_idx2 = j
            val_acc_backprop_best = target_val_accuracy_backprop

        print("Iteration done!")

plt.figure()
plt.plot(-nn_model_backprop_best.fitness_curve)
plt.grid()
plt.xlabel('Number of Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. No. of Iterations for Backpropagation (Best Model)')
plt.savefig('images/nn_train_iterations_curve_backprop_best.png')

target_test_pred_backprop = nn_model_backprop_best.predict(FEATURES_TEST)
confusion_matrix_backprop = confusion_matrix(
    TARGET_TEST, target_test_pred_backprop)

print("Average Time", np.mean(time_backprop))
print("Time:", time_backprop)
print("Test Accuracy:",
      test_acc_backprop[backprop_best_idx1][backprop_best_idx2])
print("Confusion Matrix:", confusion_matrix_backprop)

print('Backprop Completed!')

# GA
nn_model_ga_best = mlrose_hiive.NeuralNetwork(
    hidden_nodes=[16], activation='relu',
    algorithm='genetic_alg',
    max_iters=500, bias=True, is_classifier=True,
    learning_rate=0.00001, early_stopping=True,
    max_attempts=100, random_state=42, curve=True
)

for i, learning_rate in enumerate(learning_rates):
    for j, population in enumerate(populations):
        nn_model_ga = mlrose_hiive.NeuralNetwork(
            hidden_nodes=[16], activation='relu',
            algorithm='genetic_alg',
            max_iters=500, bias=True, pop_size=population, is_classifier=True,
            learning_rate=learning_rate, early_stopping=True,
            max_attempts=100, random_state=42, curve=True
        )

        start = time.time()
        nn_model_ga.fit(FEATURES_TRAIN, TARGET_TRAIN)
        end = time.time()
        time_ga_current = end - start

        target_train_pred_ga = nn_model_ga.predict(FEATURES_TRAIN)
        target_train_accuracy_ga = accuracy_score(
            TARGET_TRAIN, target_train_pred_ga)
        train_acc_ga[i][j] = target_train_accuracy_ga

        target_val_pred_ga = nn_model_ga.predict(FEATURES_VAL)
        target_val_accuracy_ga = accuracy_score(
            TARGET_VAL, target_val_pred_ga)
        val_acc_ga[i][j] = target_val_accuracy_ga

        target_test_pred_ga = nn_model_ga.predict(FEATURES_TEST)
        target_test_accuracy_ga = accuracy_score(
            TARGET_TEST, target_test_pred_ga)
        test_acc_ga[i][j] = target_test_accuracy_ga
        time_ga[i][j] = time_ga_current

        if target_val_accuracy_ga > val_acc_ga_best:
            nn_model_ga_best = nn_model_ga
            print("Learning Rate:", learning_rate)
            print("Population:", population)
            print("Time:", time_ga_current)
            ga_best_idx1 = i
            ga_best_idx2 = j
            val_acc_ga_best = target_val_accuracy_ga

        print("Iteration done!")

plt.figure()
plt.plot(nn_model_ga_best.fitness_curve[:, 0])
plt.grid()
plt.xlabel('Number of Iterations')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. No. of Iterations for GA (Best Model)')
plt.savefig('images/nn_train_iterations_curve_ga_best.png')

target_test_pred_ga = nn_model_ga_best.predict(FEATURES_TEST)
confusion_matrix_ga = confusion_matrix(TARGET_TEST, target_test_pred_ga)

print("Average Time", np.mean(time_ga))
print("Time:", time_ga)
print("Test Accuracy:", test_acc_ga[ga_best_idx1][ga_best_idx2])
print("Confusion Matrix:", confusion_matrix_ga)

print('GA Completed!')

plt.figure()
plt.bar(['RHC', 'SA', 'GA', 'Backprop'], [time_rhc[rhc_best_idx1][rhc_best_idx2], time_sa[sa_best_idx1]
        [sa_best_idx2], time_ga[ga_best_idx1][ga_best_idx2], time_backprop[backprop_best_idx1][backprop_best_idx2]])
plt.xlabel("Algorithm")
plt.ylabel("Best Time (s)")
plt.title('Best Times for Algorithms')
plt.savefig('images/nn_best_times.png')

plt.figure()
plt.bar(['RHC', 'SA', 'GA', 'Backprop'], [test_acc_rhc[rhc_best_idx1][rhc_best_idx2], test_acc_sa[sa_best_idx1]
        [sa_best_idx2], test_acc_ga[ga_best_idx1][ga_best_idx2], test_acc_backprop[backprop_best_idx1][backprop_best_idx2]])
plt.xlabel("Algorithm")
plt.ylabel("Best Score (%)")
plt.title('Best Test Score for Algorithms')
plt.ylim((0.9, 1.0))
plt.savefig('images/nn_best_test_scores.png')

plt.figure()
plt.bar(['RHC', 'SA', 'GA', 'Backprop'], [train_acc_rhc[rhc_best_idx1][rhc_best_idx2], train_acc_sa[sa_best_idx1]
        [sa_best_idx2], train_acc_ga[ga_best_idx1][ga_best_idx2], train_acc_backprop[backprop_best_idx1][backprop_best_idx2]])
plt.xlabel("Algorithm")
plt.ylabel("Best Score (%)")
plt.title('Best Train Score for Algorithms')
plt.ylim((0.9, 1.0))
plt.savefig('images/nn_best_train_scores.png')

plt.figure()
plt.bar(['RHC', 'SA', 'GA', 'Backprop'], [val_acc_rhc[rhc_best_idx1][rhc_best_idx2], val_acc_sa[sa_best_idx1]
        [sa_best_idx2], val_acc_ga[ga_best_idx1][ga_best_idx2], val_acc_backprop[backprop_best_idx1][backprop_best_idx2]])
plt.xlabel("Algorithm")
plt.ylabel("Best Score (%)")
plt.title('Best Validation Score for Algorithms')
plt.ylim((0.9, 1.0))
plt.savefig('images/nn_best_val_scores.png')

plt.figure(figsize=(12, 4))
plt.tight_layout()
plt.subplot(131)
plt.xlabel("Restarts")
plt.ylabel("Time (s)")
plt.title("RHC Analysis")
plt.errorbar(x=['2', '4', '6', '8', '10'], y=np.mean(
    time_rhc, axis=0), yerr=np.std(time_rhc, axis=0))
plt.subplot(132)
plt.xlabel("Decay")
plt.title("SA Analysis")
plt.errorbar(['GeomDecay', 'ExpDecay', 'ArithDecay'], np.mean(
    time_sa, axis=0), yerr=np.std(time_sa, axis=0))
plt.subplot(133)
plt.xlabel("Population")
plt.title("GA Analysis")
plt.errorbar(['20', '50', '100', '200', '500'], np.mean(
    time_ga, axis=0), yerr=np.std(time_ga, axis=0))
plt.savefig('images/nn_times_with_error_bars.png')

test_sizes = [0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01]
train_acc_ga_lrc = []
test_acc_ga_lrc = []
train_acc_backprop_lrc = []
test_acc_backprop_lrc = []
train_acc_sa_lrc = []
test_acc_sa_lrc = []
train_acc_rhc_lrc = []
test_acc_rhc_lrc = []

for test_size in test_sizes:
    FEATURES_TRAIN, FEATURES_TEST, TARGET_TRAIN, TARGET_TEST = train_test_split(
        features, target, test_size=test_size, random_state=69)
    nn_model_ga = mlrose_hiive.NeuralNetwork(
        hidden_nodes=[16], activation='relu',
        algorithm='genetic_alg',
        max_iters=500, bias=True, pop_size=populations[ga_best_idx2], is_classifier=True,
        learning_rate=learning_rates[ga_best_idx1], early_stopping=True,
        max_attempts=100, random_state=42, curve=True
    )
    nn_model_backprop = mlrose_hiive.NeuralNetwork(
        hidden_nodes=[16], activation='relu',
        algorithm='gradient_descent',
        max_iters=1000, bias=True, is_classifier=True,
        learning_rate=learning_rates[backprop_best_idx1], early_stopping=True,
        max_attempts=100, random_state=42, curve=True
    )
    nn_model_sa = mlrose_hiive.NeuralNetwork(
        hidden_nodes=[16], activation='relu',
        algorithm='simulated_annealing',
        max_iters=2000, bias=True, schedule=schedules[sa_best_idx2], is_classifier=True,
        learning_rate=learning_rates[sa_best_idx1], early_stopping=True,
        max_attempts=100, random_state=42, curve=True
    )
    nn_model_rhc = mlrose_hiive.NeuralNetwork(
        hidden_nodes=[16], activation='relu',
        algorithm='random_hill_climb',
        max_iters=2000, bias=True, restarts=restarts[rhc_best_idx2], is_classifier=True,
        learning_rate=learning_rates[rhc_best_idx1], early_stopping=True,
        max_attempts=100, random_state=42, curve=True
    )

    nn_model_ga.fit(FEATURES_TRAIN, TARGET_TRAIN)

    target_train_pred_ga = nn_model_ga.predict(FEATURES_TRAIN)
    target_train_accuracy_ga = accuracy_score(
        TARGET_TRAIN, target_train_pred_ga)
    train_acc_ga_lrc.append(target_train_accuracy_ga)

    target_test_pred_ga = nn_model_ga.predict(FEATURES_TEST)
    target_test_accuracy_ga = accuracy_score(TARGET_TEST, target_test_pred_ga)
    test_acc_ga_lrc.append(target_test_accuracy_ga)

    nn_model_backprop.fit(FEATURES_TRAIN, TARGET_TRAIN)

    target_train_pred_backprop = nn_model_backprop.predict(FEATURES_TRAIN)
    target_train_accuracy_backprop = accuracy_score(
        TARGET_TRAIN, target_train_pred_backprop)
    train_acc_backprop_lrc.append(target_train_accuracy_backprop)

    target_test_pred_backprop = nn_model_backprop.predict(FEATURES_TEST)
    target_test_accuracy_backprop = accuracy_score(
        TARGET_TEST, target_test_pred_backprop)
    test_acc_backprop_lrc.append(target_test_accuracy_backprop)

    nn_model_rhc.fit(FEATURES_TRAIN, TARGET_TRAIN)

    target_train_pred_rhc = nn_model_rhc.predict(FEATURES_TRAIN)
    target_train_accuracy_rhc = accuracy_score(
        TARGET_TRAIN, target_train_pred_rhc)
    train_acc_rhc_lrc.append(target_train_accuracy_rhc)

    target_test_pred_rhc = nn_model_rhc.predict(FEATURES_TEST)
    target_test_accuracy_rhc = accuracy_score(
        TARGET_TEST, target_test_pred_rhc)
    test_acc_rhc_lrc.append(target_test_accuracy_rhc)

    nn_model_sa.fit(FEATURES_TRAIN, TARGET_TRAIN)

    target_train_pred_sa = nn_model_sa.predict(FEATURES_TRAIN)
    target_train_accuracy_sa = accuracy_score(
        TARGET_TRAIN, target_train_pred_sa)
    train_acc_sa_lrc.append(target_train_accuracy_sa)

    target_test_pred_sa = nn_model_sa.predict(FEATURES_TEST)
    target_test_accuracy_sa = accuracy_score(TARGET_TEST, target_test_pred_sa)
    test_acc_sa_lrc.append(target_test_accuracy_sa)

train_sizes = [1 - test_size for test_size in test_sizes]

plt.figure()
plt.plot(train_sizes, train_acc_ga_lrc, label='Train')
plt.plot(train_sizes, test_acc_ga_lrc, label='Test')
plt.grid()
plt.legend()
plt.xlabel('Percent of Training Examples')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve for GA')
plt.savefig('images/nn_learning_curve_ga.png')

plt.figure()
plt.plot(train_sizes, train_acc_backprop_lrc, label='Train')
plt.plot(train_sizes, test_acc_backprop_lrc, label='Test')
plt.grid()
plt.legend()
plt.xlabel('Percent of Training Examples')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve for Backpropagation')
plt.savefig('images/nn_learning_curve_backprop.png')

plt.figure()
plt.plot(train_sizes, train_acc_sa_lrc, label='Train')
plt.plot(train_sizes, test_acc_sa_lrc, label='Test')
plt.grid()
plt.legend()
plt.xlabel('Percent of Training Examples')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve for SA')
plt.savefig('images/nn_learning_curve_sa.png')

plt.figure()
plt.plot(train_sizes, train_acc_rhc_lrc, label='Train')
plt.plot(train_sizes, test_acc_rhc_lrc, label='Test')
plt.grid()
plt.legend()
plt.xlabel('Percent of Training Examples')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve for RHC')
plt.savefig('images/nn_learning_curve_rhc.png')
