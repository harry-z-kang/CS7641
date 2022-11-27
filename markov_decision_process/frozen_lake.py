# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
import time
import numpy as np
import matplotlib.pyplot as plt

from gym.envs.toy_text import FrozenLakeEnv


def build_run_stat(iteration, start_time, reward, policy, value_func):
    return {
        'Reward': reward,
        'Time': time.time() - start_time,
        'Max V': np.max(value_func),
        'Mean V': np.mean(value_func),
        'Iteration': iteration,
        'Value': value_func.copy(),
        'Policy': policy.copy()
    }


def policy_iteration(env: FrozenLakeEnv, gamma=1.0, max_iters=3000) -> list:
    run_stats = []
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)

    start_time = time.time()
    for i in range(max_iters):
        current_policy_value_func = compute_policy_value_func_iterative(
            env, policy, gamma)
        new_policy = extract_policy(env, current_policy_value_func, gamma)

        run_stats.append(build_run_stat(
            i + 1, start_time, np.max(current_policy_value_func), new_policy, current_policy_value_func))

        if np.all(policy == new_policy):
            break

        policy = new_policy

    return run_stats


def compute_policy_value_func_iterative(env: FrozenLakeEnv, policy: np.ndarray[np.int32], gamma: float) -> np.ndarray[np.float64]:
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    eps = 1e-5
    value_func = np.zeros(env.observation_space.n)

    while True:
        prev_value_func = np.copy(value_func)

        for state in range(env.observation_space.n):
            action = policy[state]
            value_func[state] = sum([
                # Bellman Equation: P * (R + gamma * V)
                p * (r + gamma * prev_value_func[nxt_state]) for p, nxt_state, r, _ in env.P[state][action]
            ])

        if np.sum((np.fabs(prev_value_func - value_func))) <= eps:
            break

    return value_func


def value_iteration(env: FrozenLakeEnv, gamma=1.0, max_iters=10000) -> list:
    eps = 1e-20
    run_stats = []
    value_func = np.zeros(env.observation_space.n)

    start_time = time.time()
    for i in range(max_iters):
        prev_value_func = np.copy(value_func)

        for state in range(env.observation_space.n):
            q_sa = [sum([
                # Bellman Equation: P * (R + gamma * V)
                p * (r + gamma * prev_value_func[nxt_state]) for p, nxt_state, r, _ in env.P[state][action]
            ]) for action in range(env.action_space.n)]
            value_func[state] = max(q_sa)

        run_stats.append(build_run_stat(i + 1, start_time, np.max(value_func),
                         extract_policy(env, value_func, gamma), value_func))

        if np.sum(np.fabs(prev_value_func - value_func)) <= eps:
            break

    return run_stats


def extract_policy(env: FrozenLakeEnv, value_func: np.ndarray[np.float64], gamma: float) -> np.ndarray[np.int32]:
    policy = np.zeros(env.observation_space.n, dtype=np.int32)

    for state in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):
            q_sa[action] = sum([
                # Bellman Equation: P * (R + gamma * V)
                p * (r + gamma * value_func[nxt_state]) for p, nxt_state, r, _ in  env.P[state][action]
            ])

        policy[state] = np.argmax(q_sa)

    return policy


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
    font_size = 'x-large'
    if policy.shape[1] > 16:
        font_size = 'small'
    plt.title(title)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            y = policy.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                    horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    plt.savefig(f'./images/frozen_lake/{title}.png')
    plt.close()


def colors_lake():
    return {
        b'S': 'red',
        b'F': 'skyblue',
        b'H': 'black',
        b'G': 'green',
    }


def directions_lake():
    return {
        3: '⬆',
        2: '➡',
        1: '⬇',
        0: '⬅'
    }


# https://gist.github.com/jojonki/6291f8c3b19799bc2f6d5279232553d7
# Q learning params
ALPHA = 0.5  # learning rate
GAMMA = 0.999  # reward discount


def q_learning(env: FrozenLakeEnv, map_size: int, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999, alpha=ALPHA, gamma=GAMMA, n_iter=1000000) -> tuple[list, list]:
    run_stats = []
    total_reward = 0.0
    Q = np.zeros(map_size * map_size * 4) \
        .reshape(map_size * map_size, 4) \
        .astype(np.float64)

    start_time = time.time()
    for t in range(n_iter):
        state = env.reset()[0]
        for _ in range(20000):
            greedy_prob = np.random.random()
            if greedy_prob < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            q_target_action = np.max(Q[next_state])
            # Q <- Q + a(Q' - Q) <=> Q <- (1-a)Q + a(Q')
            Q[state][action] += alpha * \
                (reward + gamma * q_target_action - Q[state, action])

            if done or t == 20000 - 1:
                break

            state = next_state

        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        run_stats.append(build_run_stat(
            t + 1, start_time, reward,
            [np.argmax(Q[state]) for state in range(map_size * map_size)],
            [np.max(Q[state]) for state in range(map_size * map_size)]
        ))

    return run_stats, Q


def run_policy_iteration(env: FrozenLakeEnv, map_size: int):
    run_stats_array = []

    gamma_array = [0.2, 0.4, 0.8, 0.9, 0.95, 0.99, 1]
    for g in gamma_array:
        run_stats = policy_iteration(env, gamma=g)

        run_stats_array.append(run_stats)

        with open(f"./logs/frozen_lake/PI_Policy_Migration_{map_size}_{g}.txt", "w") as f:
            f.write("Different Discount Factor:\n")
            f.write(f"\t{g}: {run_stats[-1]['Policy']}\n")
            f.write(f"\tTime (s): {run_stats[-1]['Time']}\n")
            f.write(f"\tMean Value: {run_stats[-1]['Mean V']}\n")
            f.write(
                f"\tIterations to converge: {run_stats[-1]['Iteration']}\n\n")
            for i, r in enumerate(run_stats):
                f.write(f"Iteration {i}:")
                f.write(f"\t{r['Policy'].reshape(map_size, map_size)}\n")
                f.write(f"\t{r['Value']}\n")

    plt.title('PI, Discount Factor vs. Execution Time')
    plt.xlabel('Gamma')
    plt.ylabel('Execution Time (s)')
    plt.bar(np.arange(len(gamma_array)), [
            r[-1]["Time"] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0],
                  label_type="edge", fmt="%.2f", padding=1.5)
    plt.xticks(np.arange(len(gamma_array)), gamma_array)
    plt.savefig(f"./images/frozen_lake/PI_Gamma_ExecutionTime_{map_size}.png")
    plt.show()

    plt.title('Policy Iteration, Reward Analysis')
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.plot(gamma_array, [r[-1]['Mean V'] for r in run_stats_array])
    plt.grid()
    plt.savefig(f"./images/frozen_lake/PI_AverageRewards_{map_size}.png")
    plt.show()

    plt.title('PI, Discount Factor vs. Iterations to Converge')
    plt.xlabel('Gamma')
    plt.ylabel('Iterations to Converge')
    plt.bar(np.arange(len(gamma_array)), [
            r[-1]['Iteration'] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0], label_type="edge", padding=1.5)
    plt.xticks(np.arange(len(gamma_array)), gamma_array)
    plt.savefig(f"./images/frozen_lake/PI_Gamma_Iteration_{map_size}.png")
    plt.show()

    for g, run_stat in zip(gamma_array[2:], run_stats_array[2:]):
        plt.figure()
        plt.title(f'PI, Iteration vs. Reward, Gamma={g}')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        for s in range(env.observation_space.n):
            value_array = []
            for r in run_stat:
                value_array.append(r['Value'][s])
            plt.plot(np.arange(len(run_stat)), value_array, label=f"State {s}")
        mean_value_array = []
        for iter, r in enumerate(run_stat):
            mean_value_array.append(r['Mean V'])
            plot_policy_map(f"PI_{map_size}_{g}_{iter}", r['Policy'].reshape(map_size, map_size), env.desc, colors_lake(), directions_lake())
        plt.plot(np.arange(len(run_stat)),
                 mean_value_array, label="Mean Value")
        plt.grid()
        if env.observation_space.n <= 10:
            plt.legend(loc='upper right')
        plt.savefig(
            f"./images/frozen_lake/PI_Iteration_Reward_{map_size}_{g}.png")
    

def run_value_iteration(env: FrozenLakeEnv, map_size: int):
    run_stats_array = []

    gamma_array = [0.2, 0.4, 0.8, 0.9, 0.95, 0.99, 1]
    for g in gamma_array:
        run_stats = value_iteration(env, gamma=g)

        run_stats_array.append(run_stats)

        with open(f"./logs/frozen_lake/VI_Policy_Migration_{map_size}_{g}.txt", "w") as f:
            f.write("Different Discount Factor:\n")
            f.write(f"\t{g}: {run_stats[-1]['Policy']}\n")
            f.write(f"\tTime (ms): {run_stats[-1]['Time'] * 1000}\n")
            f.write(f"\tMean Value: {run_stats[-1]['Mean V']}\n")
            f.write(
                f"\tIterations to converge: {run_stats[-1]['Iteration']}\n\n")
            for i, r in enumerate(run_stats):
                f.write(f"Iteration {i}:\n")
                f.write(f"\t{r['Policy'].reshape(map_size, map_size)}\n")
                f.write(f"\t{r['Value']}\n")

    plt.title('VI, Discount Factor vs. Execution Time')
    plt.xlabel('Gamma')
    plt.ylabel('Execution Time (s)')
    plt.bar(np.arange(len(gamma_array)), [
            r[-1]["Time"] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0],
                  label_type="edge", fmt="%.2f", padding=1.5)
    plt.xticks(np.arange(len(gamma_array)), gamma_array)
    plt.savefig(f"./images/frozen_lake/VI_Gamma_ExecutionTime_{map_size}.png")
    plt.show()

    plt.plot(gamma_array, [r[-1]['Mean V'] for r in run_stats_array])
    plt.title('VI, Discount Factor vs. Average Reward')
    plt.xlabel('Gamma')
    plt.ylabel('Average Reward')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/VI_Gamma_AverageReward_{map_size}.png")
    plt.show()

    plt.title('VI, Discount Factor vs. Iterations to Converge')
    plt.xlabel('Gamma')
    plt.ylabel('Iterations to Converge')
    plt.bar(np.arange(len(gamma_array)), [
            r[-1]['Iteration'] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0], label_type="edge", padding=1.5)
    plt.xticks(np.arange(len(gamma_array)), gamma_array)
    plt.savefig(f"./images/frozen_lake/VI_Gamma_Iteration_{map_size}.png")
    plt.show()

    for g, run_stat in zip(gamma_array[2:], run_stats_array[2:]):
        plt.figure()
        plt.title(f'VI, Iteration vs. Reward, Gamma={g}')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        for s in range(env.observation_space.n):
            value_array = []
            for r in run_stat:
                value_array.append(r['Value'][s])
            plt.plot(np.arange(len(run_stat)), value_array, label=f"State {s}")
        mean_value_array = []
        for iter, r in enumerate(run_stat):
            mean_value_array.append(r['Mean V'])
            if iter % 300 == 0 or iter == len(run_stat) - 1:
                plot_policy_map(f"VI_{map_size}_{g}_{iter}", r['Policy'].reshape(map_size, map_size), env.desc, colors_lake(), directions_lake())
        plt.plot(np.arange(len(run_stat)),
                 mean_value_array, label="Mean Value")
        plt.grid()
        if env.observation_space.n <= 10:
            plt.legend(loc='upper right')
        plt.savefig(
            f"./images/frozen_lake/VI_Iteration_Reward_{map_size}_{g}.png")


def run_q_learning(env: FrozenLakeEnv, map_size: int):
    run_stats_array = []

    epsilon_array = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]
    for e in epsilon_array:
        run_stats, Q = q_learning(env, gamma=0.95, epsilon=e,
                                  epsilon_decay=0.999, n_iter=50000, alpha=0.5, map_size=map_size)

        run_stats_array.append(run_stats)

        with open(f"./logs/frozen_lake/QL_Epsilon_Policy_Migration_{map_size}_{e}.txt", "w") as f:
            f.write("Different Greedy Rate:\n")
            f.write(f"\t{e}: {run_stats[-1]['Policy']}\n")
            f.write(f"\tTime (s): {run_stats[-1]['Time']}\n")
            f.write(f"\tMean Value: {run_stats[-1]['Mean V']}\n")
            f.write(f"\tLearned Q Table: \n{Q}\n\n")
            for i, r in enumerate(run_stats):
                f.write(f"Iteration {i}:")
                f.write(f"{r['Policy']}\n")
                f.write(f"{r['Value']}\n")

    plt.figure()
    plt.plot(epsilon_array, [r[-1]['Mean V'] for r in run_stats_array])
    plt.xlabel('Epsilon')
    plt.ylabel('Average Reward')
    plt.title('QL, Greedy Rate vs. Average Reward')
    plt.grid()
    plt.savefig(
        f"./images/frozen_lake/QL_Epsilon_AverageReward_{map_size}.png")
    plt.show()

    plt.figure()
    plt.title('QL, Greedy Rate vs. Execution Time')
    plt.xlabel('Epsilon')
    plt.ylabel('Execution Time (ms)')
    plt.bar(np.arange(len(epsilon_array)), [
            r[-1]['Time'] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0],
                  label_type="edge", fmt="%.2f", padding=1.5)
    plt.xticks(np.arange(len(epsilon_array)), epsilon_array)
    plt.savefig(
        f"./images/frozen_lake/QL_Epsilon_ExecutionTime_{map_size}.png")
    plt.show()

    for e, run_stat in zip(epsilon_array, run_stats_array):
        plt.figure()
        plt.title(f'QL, Epsilon vs. Reward, Epsilon={e}')
        plt.xlabel('Epsilon')
        plt.ylabel('Reward')
        for s in range(env.observation_space.n):
            value_array = []
            for r in run_stat:
                value_array.append(r['Value'][s])
            plt.plot(np.arange(len(run_stat)), value_array, label=f"State {s}")
        mean_value_array = []
        for iter, r in enumerate(run_stat):
            mean_value_array.append(r['Mean V'])
            if iter % 1000 == 0 or iter == len(run_stat) - 1:
                plot_policy_map(f"QL_Epsilon_{map_size}_{e}_{iter}", np.array(r['Policy']).reshape(map_size, map_size), env.desc, colors_lake(), directions_lake())
        plt.plot(np.arange(len(run_stat)),
                 mean_value_array, label="Mean Value")
        plt.grid()
        if env.observation_space.n <= 10:
            plt.legend(loc='upper right')
        plt.savefig(
            f"./images/frozen_lake/QL_Epsilon_Iteration_Reward_{map_size}_{e}.png")

    run_stats_array = []

    gamma_array = [0.2, 0.4, 0.8, 0.9, 0.95, 0.99, 1]
    for g in gamma_array:
        run_stats, Q = q_learning(
            env, gamma=g, epsilon=0.95, epsilon_decay=0.999, n_iter=50000, alpha=1, map_size=map_size)

        run_stats_array.append(run_stats)

        with open(f"./logs/frozen_lake/QL_Gamma_Policy_Migration_{map_size}_{g}.txt", "w") as f:
            f.write("Different Discount Factor:\n")
            f.write(f"\t{g}: {run_stats[-1]['Policy']}\n")
            f.write(f"\tTime (s): {run_stats[-1]['Time']}\n")
            f.write(f"\tMean Value: {run_stats[-1]['Mean V']}\n")
            f.write(f"\tLearned Q Table: \n{Q}\n\n")
            for i, r in enumerate(run_stats):
                f.write(f"Iteration {i}:")
                f.write(f"{r['Policy']}\n")
                f.write(f"{r['Value']}\n")

    plt.plot(gamma_array, [r[-1]['Mean V'] for r in run_stats_array])
    plt.title('QL, Discount Factor vs. Average Reward')
    plt.xlabel('Gamma')
    plt.ylabel('Average Rewards')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/QL_Gamma_AverageReward_{map_size}.png")
    plt.show()

    plt.plot(gamma_array, [r[-1]['Time'] for r in run_stats_array])
    plt.title('QL, Discount Factor vs. Execution Time')
    plt.xlabel('Gamma')
    plt.ylabel('Execution Time (ms)')
    plt.bar(np.arange(len(gamma_array)), [
            r[-1]['Time'] for r in run_stats_array], width=0.5, align='center')
    plt.bar_label(plt.gca().containers[0],
                  label_type="edge", fmt="%.2f", padding=1.5)
    plt.savefig(f"./images/frozen_lake/QL_Gamma_ExecutionTime_{map_size}.png")
    plt.show()

    for g, run_stat in zip(gamma_array[2:], run_stats_array[2:]):
        plt.figure()
        plt.title(f'QL, Iteration vs. Reward, Gamma={g}')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        for s in range(env.observation_space.n):
            value_array = []
            for r in run_stat:
                value_array.append(r['Value'][s])
            plt.plot(np.arange(len(run_stat)), value_array, label=f"State {s}")
        mean_value_array = []
        for iter, r in enumerate(run_stat):
            mean_value_array.append(r['Mean V'])
            if iter % 1000 == 0 or iter == len(run_stat) - 1:
                plot_policy_map(f"QL_Gamma_{map_size}_{g}_{iter}", np.array(r['Policy']).reshape(map_size, map_size), env.desc, colors_lake(), directions_lake())
        plt.plot(np.arange(len(run_stat)),
                 mean_value_array, label="Mean Value")
        plt.grid()
        if env.observation_space.n <= 10:
            plt.legend(loc='upper right')
        plt.savefig(
            f"./images/frozen_lake/QL_Gamma_Iteration_Reward_{map_size}_{g}.png")
