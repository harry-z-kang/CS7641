import numpy as np
import matplotlib.pyplot as plt

from hiive.mdptoolbox import example, mdp


def run_policy_iteration(S, r1, r2):
    transition_matrix, reward_matrix = example.forest(S=S, r1=r1, r2=r2)

    run_stats_array = []

    gamma_array = [(i + 0.5) / 10 for i in range(0, 10)]
    for g in gamma_array:
        pi = mdp.PolicyIteration(transition_matrix, reward_matrix, g)
        pi.run()

        run_stats_array.append(pi.run_stats)

        with open(f"./logs/forest/PI_Policy_Migration_{S}_{r1}_{r2}_{g}.txt", "w") as f:
            f.write("Different Discount Factor:\n")
            f.write(f"\t{g}: {pi.run_stats[-1]['Policy']}\n")
            f.write(f"\tTime (ms): {pi.run_stats[-1]['Time'] * 1000}\n")
            f.write(f"\tMean Value: {pi.run_stats[-1]['Mean V']}\n")
            f.write(f"\tIterations to converge: {pi.run_stats[-1]['Iteration']}\n\n")
            for i, r in enumerate(pi.run_stats):
                f.write(f"Iteration {i}:")
                f.write(f"\t{r['Policy']}\n")
                f.write(f"\t{r['Value']}\n")

    plt.title('PI, Discount Factor vs. Execution Time')
    plt.xlabel('Gamma')
    plt.ylabel('Execution Time (ms)')
    plt.bar(np.arange(len(gamma_array)), [
            r[-1]["Time"] * 1000 for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0],
                  label_type="edge", fmt="%.2f", padding=1.5)
    plt.xticks(np.arange(len(gamma_array)), gamma_array)
    plt.savefig(f"./images/forest/PI_Gamma_ExecutionTime_{S}_{r1}_{r2}.png")
    plt.show()

    plt.title('PI, Discount Factor vs. Average Reward')
    plt.xlabel('Gamma')
    plt.ylabel('Average Reward')
    plt.plot(gamma_array, [r[-1]['Mean V'] for r in run_stats_array])
    plt.grid()
    plt.savefig(f"./images/forest/PI_Gamma_AverageReward_{S}_{r1}_{r2}.png")
    plt.show()

    plt.title('PI, Discount Factor vs. Iterations to Converge')
    plt.xlabel('Gamma')
    plt.ylabel('Iterations to Converge')
    plt.bar(np.arange(len(gamma_array)), [
            r[-1]['Iteration'] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0], label_type="edge", padding=1.5)
    plt.xticks(np.arange(len(gamma_array)), gamma_array)
    plt.savefig(f"./images/forest/PI_Gamma_Iteration_{S}_{r1}_{r2}.png")
    plt.show()

    for g, run_stat in zip(gamma_array[5:], run_stats_array[5:]): 
        plt.figure()
        plt.title(f'PI, Iteration vs. Reward, Gamma={g}')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        for s in range(S):
            value_array = []
            for r in run_stat:
                value_array.append(r['Value'][s])
            plt.plot(np.arange(len(run_stat)), value_array, label=f"State {s}")
        mean_value_array = []
        for r in run_stat:
            mean_value_array.append(r['Mean V'])
        plt.plot(np.arange(len(run_stat)),
                 mean_value_array, label="Mean Value")
        plt.grid()
        if S <= 10:
          plt.legend(loc='upper right')
        plt.savefig(
            f"./images/forest/PI_Iteration_Reward_{S}_{r1}_{r2}_{g}.png")


def run_value_iteration(S, r1, r2):
    transition_matrix, reward_matrix = example.forest(S=S, r1=r1, r2=r2)

    run_stats_array = []

    gamma_array = [(i + 0.5) / 10 for i in range(0, 10)]
    for g in gamma_array:
        vi = mdp.ValueIteration(transition_matrix, reward_matrix, g)
        vi.run()

        run_stats_array.append(vi.run_stats)

        with open(f"./logs/forest/VI_Policy_Migration_{S}_{r1}_{r2}_{g}.txt", "w") as f:
            f.write("Different Discount Factor:\n")
            f.write(f"\t{g}: {vi.run_stats[-1]['Policy']}\n")
            f.write(f"\tTime (ms): {vi.run_stats[-1]['Time'] * 1000}\n")
            f.write(f"\tMean Value: {vi.run_stats[-1]['Mean V']}\n")
            f.write(f"\tIterations to converge: {vi.run_stats[-1]['Iteration']}\n\n")
            for i, r in enumerate(vi.run_stats):
                f.write(f"Iteration {i}:\n")
                f.write(f"\t{r['Policy']}\n")
                f.write(f"\t{r['Value']}\n")


    plt.title('VI, Discount Factor vs. Execution Time')
    plt.xlabel('Gamma')
    plt.ylabel('Execution Time (ms)')
    plt.bar(np.arange(len(gamma_array)), [
            r[-1]["Time"] * 1000 for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0],
                  label_type="edge", fmt="%.2f", padding=1.5)
    plt.xticks(np.arange(len(gamma_array)), gamma_array)
    plt.savefig(f"./images/forest/VI_Gamma_ExecutionTime_{S}_{r1}_{r2}.png")
    plt.show()

    plt.plot(gamma_array, [r[-1]['Mean V'] for r in run_stats_array])
    plt.title('VI, Discount Factor vs. Average Reward')
    plt.xlabel('Gamma')
    plt.ylabel('Average Reward')
    plt.grid()
    plt.savefig(f"./images/forest/VI_Gamma_AverageReward_{S}_{r1}_{r2}.png")
    plt.show()

    plt.title('VI, Discount Factor vs. Iterations to Converge')
    plt.xlabel('Gamma')
    plt.ylabel('Iterations to Converge')
    plt.bar(np.arange(len(gamma_array)), [r[-1]['Iteration'] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0], label_type="edge", padding=1.5)
    plt.xticks(np.arange(len(gamma_array)), gamma_array)
    plt.savefig(f"./images/forest/VI_Gamma_Iteration_{S}_{r1}_{r2}.png")
    plt.show()

    for g, run_stat in zip(gamma_array[5:], run_stats_array[5:]):
        plt.figure()
        plt.title(f'VI, Iteration vs. Reward, Gamma={g}')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        for s in range(S):
            value_array = []
            for r in run_stat:
                value_array.append(r['Value'][s])
            plt.plot(np.arange(len(run_stat)), value_array, label=f"State {s}")
        mean_value_array = []
        for r in run_stat:
            mean_value_array.append(r['Mean V'])
        plt.plot(np.arange(len(run_stat)),
                 mean_value_array, label="Mean Value")
        plt.grid()
        if S <= 10:
          plt.legend(loc='upper right')
        plt.savefig(
            f"./images/forest/VI_Iteration_Reward_{S}_{r1}_{r2}_{g}.png")


def run_q_learning(S, r1, r2):
    transition_matrix, reward_matrix = example.forest(S=S, r1=r1, r2=r2)

    run_stats_array = []

    epsilon_array = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]
    for e in epsilon_array:
        ql = mdp.QLearning(transition_matrix, reward_matrix,
                           0.95, epsilon=e, epsilon_decay=0.999, n_iter=1000000, alpha=1)
        ql.run()

        run_stats_array.append(ql.run_stats)

        with open(f"./logs/forest/QL_Epsilon_Policy_Migration_{S}_{r1}_{r2}_{e}.txt", "w") as f:
            f.write("Different Greedy Rate:\n")
            f.write(f"\t{e}: {ql.run_stats[-1]['Policy']}\n")
            f.write(f"\tTime (s): {ql.run_stats[-1]['Time']}\n")
            f.write(f"\tMean Value: {ql.run_stats[-1]['Mean V']}\n")
            f.write(f"\tLearned Q Table: \n{ql.Q}\n\n")
            for i, r in enumerate(ql.run_stats):
                f.write(f"Iteration {i}:")
                f.write(f"{r['Policy']}")
                f.write(f"{r['Value']}\n")


    plt.plot(epsilon_array, [r[-1]['Mean V'] for r in run_stats_array])
    plt.xlabel('Epsilon')
    plt.ylabel('Average Reward')
    plt.title('QL, Greedy Rate vs. Average Reward')
    plt.grid()
    plt.savefig(f"./images/forest/QL_Epsilon_AverageReward_{S}_{r1}_{r2}.png")
    plt.show()

    plt.plot(epsilon_array, [r[-1]['Time'] for r in run_stats_array])
    plt.title('QL, Greedy Rate vs. Execution Time')
    plt.xlabel('Epsilon')
    plt.ylabel('Execution Time (ms)')
    plt.bar(np.arange(len(epsilon_array)), [
            r[-1]['Time'] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0],
                  label_type="edge", fmt="%.2f", padding=1.5)
    plt.xticks(np.arange(len(epsilon_array)), epsilon_array)
    plt.savefig(f"./images/forest/QL_Epsilon_ExecutionTime_{S}_{r1}_{r2}.png")
    plt.show()

    for e, run_stat in zip(epsilon_array, run_stats_array):
        plt.figure()
        plt.title(f'QL, Epsilon vs. Reward, Epsilon={e}')
        plt.xlabel('Epsilon')
        plt.ylabel('Reward')
        for s in range(S):
            value_array = []
            for r in run_stat:
                value_array.append(r['Value'][s])
            plt.plot(np.arange(len(run_stat)), value_array, label=f"State {s}")
        mean_value_array = []
        for r in run_stat:
            mean_value_array.append(r['Mean V'])
        plt.plot(np.arange(len(run_stat)),
                 mean_value_array, label="Mean Value")
        plt.grid()
        if S <= 10:
          plt.legend(loc='upper right')
        plt.legend(loc='upper right')
        plt.savefig(
            f"./images/forest/QL_Epsilon_Iteration_Reward_{S}_{r1}_{r2}_{e}.png")

    transition_matrix, reward_matrix = example.forest(S=S, r1=r1, r2=r2)

    run_stats_array = []

    gamma_array = [(i + 0.5) / 10 for i in range(0, 10)]
    for e in gamma_array:
        ql = mdp.QLearning(transition_matrix, reward_matrix,
                           e, epsilon=0.05, epsilon_decay=0.999, n_iter=1000000, alpha=1)
        ql.run()

        run_stats_array.append(ql.run_stats)

        with open(f"./logs/forest/QL_Gamma_Policy_Migration_{S}_{r1}_{r2}_{e}.txt", "w") as f:
            f.write("Different Discount Factor:\n")
            f.write(f"\t{e}: {ql.run_stats[-1]['Policy']}\n")
            f.write(f"\tTime (s): {ql.run_stats[-1]['Time']}\n")
            f.write(f"\tMean Value: {ql.run_stats[-1]['Mean V']}\n")
            f.write(f"\tLearned Q Table: \n{ql.Q}\n\n")
            for i, r in enumerate(ql.run_stats):
                f.write(f"Iteration {i}:")
                f.write(f"{r['Policy']}")
                f.write(f"{r['Value']}\n")


    plt.plot(gamma_array, [r[-1]['Mean V'] for r in run_stats_array])
    plt.title('QL, Discount Factor vs. Average Reward')
    plt.xlabel('Gamma')
    plt.ylabel('Average Rewards')
    plt.grid()
    plt.savefig(f"./images/forest/QL_Gamma_AverageReward_{S}_{r1}_{r2}.png")
    plt.show()

    plt.plot(gamma_array, [r[-1]['Time'] for r in run_stats_array])
    plt.title('QL, Discount Factor vs. Execution Time')
    plt.xlabel('Gamma')
    plt.ylabel('Execution Time (ms)')
    plt.bar(np.arange(len(gamma_array)), [
            r[-1]['Time'] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0],
                  label_type="edge", fmt="%.2f", padding=1.5)
    plt.savefig(f"./images/forest/QL_Gamma_ExecutionTime_{S}_{r1}_{r2}.png")
    plt.show()

    for e, run_stat in zip(gamma_array[5:], run_stats_array[5:]):
        plt.figure()
        plt.title(f'QL, Iteration vs. Reward, Gamma={e}')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        for s in range(S):
            value_array = []
            for r in run_stat:
                value_array.append(r['Value'][s])
            plt.plot(np.arange(len(run_stat)), value_array, label=f"State {s}")
        mean_value_array = []
        for r in run_stat:
            mean_value_array.append(r['Mean V'])
        plt.plot(np.arange(len(run_stat)),
                 mean_value_array, label="Mean Value")
        plt.grid()
        if S <= 10:
          plt.legend(loc='upper right')
        plt.savefig(
            f"./images/forest/QL_Gamma_Iteration_Reward_{S}_{r1}_{r2}_{e}.png")
