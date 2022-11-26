# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
import time
import numpy as np
import matplotlib.pyplot as plt


def policy_iteration(env, gamma, map_size, max_iters=3000):
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)
    desc = env.unwrapped.desc
    for i in range(max_iters):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            k = i+1
            break
        policy = new_policy
        k = 1+1
        if i % 50 == 0:
            print(i)
    plot_policy_map(
        f'PI Policy Map, Iteration {i}, Gamma: {gamma}',
        policy.reshape(map_size, map_size),
        desc, colors_lake(), directions_lake()
    )
    return policy, k


def value_iteration(env, map_size, gamma=1.0, max_iterations=10000):
    v = np.zeros(env.observation_space.n)  # initialize value-function
    desc = env.unwrapped.desc
    eps = 1e-20
    for i in range(max_iterations):
        if i % 500 == 0:
            print(i)
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            q_sa = [sum(
                [p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]
            )for a in range(env.action_space.n)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            k = i+1
            break
        k = i+1
    policy = extract_policy(env, v, gamma)
    plot_policy_map(
        f'VI Policy Map, Iteration {i}, Gamma: {gamma}',
        policy.reshape(map_size, map_size),
        desc, colors_lake(), directions_lake()
    )
    return v, k


def run_episode(env, policy, gamma, render=True):
    obs = env.reset()[0]
    total_reward = 0
    step_idx = 0
    start = time.time()
    while True:
        if render:
            env.render()
        obs, reward, done, _, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done or time.time() - start > 0.01:
            break
    return total_reward


def extract_policy(env, v, gamma):
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum(
                [p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]]
            )
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma):
    v = np.zeros(env.observation_space.n)
    eps = 1e-5
    while True:
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_])
                       for p, s_, r, is_done in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v


def evaluate_policy(env, policy, gamma, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(
        0, policy.shape[1]), ylim=(0, policy.shape[0]))
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

            text = ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, policy.shape[1]))
    plt.ylim((0, policy.shape[0]))
    plt.tight_layout()
    plt.show()
    # plt.savefig([title, '.png'])
    plt.close()

    return (plt)


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
ALPHA = 0.1  # learning rate
GAMMA = 0.999  # reward discount
LEARNING_COUNT = 50000
TEST_COUNT = 1000

TURN_LIMIT = 20000


class Agent:
    def __init__(self, env, map_size):
        self.env = env
        self.episode_reward = 0.0
        self.epsilon = 1.0
        self.q_val = np.zeros(
            map_size * map_size * 4).reshape(map_size*map_size, 4).astype(np.float32)
        # exploartion decreasing decay for exponential decreasing
        self.epsilon_decay = 0.9999
        # minimum of exploration proba
        self.epsilon_min = 0.01

    def learn(self):
        state = self.env.reset()[0]
        total_reward = 0.0
        # self.env.render()

        for t in range(TURN_LIMIT):
            pn = np.random.random()
            if pn < self.epsilon:
                act = self.env.action_space.sample()  # random
            else:
                act = self.q_val[state].argmax()
            next_state, reward, done, _, _ = self.env.step(act)
            total_reward += reward
            q_next_max = np.max(self.q_val[next_state])
            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')
            self.q_val[state][act] = (1 - ALPHA) * self.q_val[state][act] + ALPHA * (
                reward + GAMMA * q_next_max - self.q_val[state, act])
            # self.env.render()
            if done or t == TURN_LIMIT - 1:
                return total_reward
            else:
                state = next_state
        self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon)

    def test(self):
        state = self.env.reset()[0]
        total_reward = 0.0
        for t in range(TURN_LIMIT):
            act = np.argmax(self.q_val[state])
            next_state, reward, done, _, _ = self.env.step(act)
            total_reward += reward
            if done or t == TURN_LIMIT - 1:
                return total_reward
            else:
                state = next_state
        return 0.0  # over limit


def run_policy_iteration(env, map_size):
    time_array = [0] * 7
    gamma_arr = [0] * 7
    iters = [0] * 7
    list_scores = [0] * 7

    for i, gamma in enumerate([0.2, 0.4, 0.8, 0.9, 0.95, 0.99, 1]):
        st = time.time()
        best_policy, k = policy_iteration(env, gamma=gamma, map_size=map_size)
        scores = evaluate_policy(env, best_policy, gamma=gamma)
        print(scores)
        end = time.time()
        gamma_arr[i] = gamma
        list_scores[i] = np.mean(scores)
        iters[i] = k
        time_array[i] = end-st

    print(gamma_arr)
    print(time_array)
    print(list_scores)
    print(iters)
    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Policy Iteration, Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/PI_ExecutionTime_{map_size}.png")
    plt.show()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Policy Iteration, Reward Analysis')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/PI_AverageRewards_{map_size}.png")
    plt.show()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title('Policy Iteration, Convergence Analysis')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/PI_Convergence_{map_size}.png")
    plt.show()
    
    list_scores = []
    policy = []
    time_array = []
    iters = [100, 200, 500, 1000, 2000, 5000, 10000]
    for iter in iters:
        st = time.time()
        best_policy, k = policy_iteration(env, gamma=0.5, map_size=map_size, max_iters=iter)
        scores = evaluate_policy(env, best_policy, gamma=0.5)
        end = time.time()
        print(scores)
        list_scores.append(np.mean(scores))
        time_array.append(end - st)

    print(f"Different iterations: {policy}")
    plt.plot(iters, list_scores)
    plt.xlabel('Iterations')
    plt.ylabel('Average Rewards')
    plt.title('Iteration vs Average Rewards Analysis')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/PI_Iterations_AverageRewards_{map_size}.png")
    plt.show()
    
    plt.plot(iters, time_array)
    plt.xlabel('Iterations')
    plt.ylabel('Execution Time')
    plt.title('Iteration vs Execution Time Analysis')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/PI_Iterations_ExecutionTime_{map_size}.png")
    plt.show()


def run_value_iteration(env, map_size):
    time_array = [0] * 7
    gamma_arr = [0] * 7
    iters = [0] * 7
    list_scores = [0] * 7

    for i, gamma in enumerate([0.2, 0.4, 0.8, 0.9, 0.95, 0.99, 1]):
        st = time.time()
        optimal_v, k = value_iteration(env, gamma=gamma, map_size=map_size)
        policy = extract_policy(env, optimal_v, gamma)
        scores = evaluate_policy(env, policy, gamma=gamma)
        print(scores)
        end = time.time()
        gamma_arr[i] = gamma
        list_scores[i] = np.mean(scores)
        iters[i] = k
        time_array[i] = end-st

    print(gamma_arr)
    print(time_array)
    print(list_scores)
    print(iters)
    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Value Iteration, Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/VI_ExecutionTime_{map_size}.png")
    plt.show()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Value Iteration, Reward Analysis')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/VI_AverageRewards_{map_size}.png")
    plt.show()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title('Value Iteration, Convergence Analysis')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/VI_Convergence_{map_size}.png")
    plt.show()
    
    list_scores = []
    policy = []
    time_array = []
    iters = [100, 200, 500, 1000, 2000, 5000, 10000]
    for iter in iters:
        st = time.time()
        optimal_v, k = value_iteration(env, gamma=0.5, map_size=map_size, max_iterations=iter)
        policy = extract_policy(env, optimal_v, 0.5)
        scores = evaluate_policy(env, policy, gamma=0.5)
        end = time.time()
        print(scores)
        list_scores.append(np.mean(scores))
        time_array.append(end-st)

    print(f"Different iterations: {policy}")
    plt.plot(iters, list_scores)
    plt.xlabel('Iterations')
    plt.ylabel('Average Rewards')
    plt.title('Iteration vs Average Rewards Analysis')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/VI_Iterations_AverageRewards_{map_size}.png")
    plt.show()
    
    plt.plot(iters, time_array)
    plt.xlabel('Iterations')
    plt.ylabel('Execution Time')
    plt.title('Iteration vs Execution Time Analysis')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/VI_Iterations_ExecutionTime_{map_size}.png")
    plt.show()


def run_q_learning(env, map_size):
    time_array = [0] * 7
    gamma_arr = [0] * 7
    list_scores = [0] * 7
    list_scores_test = [0] * 7
    policy = []

    desc = env.unwrapped.desc

    for j, gamma in enumerate([0.2, 0.4, 0.8, 0.9, 0.95, 0.99, 1]):
        GAMMA = gamma
        agent = Agent(env, map_size)
        print("###### LEARNING #####")
        reward_total = 0.0
        st = time.time()
        for i in range(LEARNING_COUNT):
            reward = agent.learn()
            reward_total += reward
            print(reward)
            if i % 10000 == 0:
                print(i)
        end = time.time()
        print("gamma         : {}".format(GAMMA))
        print("episodes      : {}".format(LEARNING_COUNT))
        print("total reward  : {}".format(reward_total))
        print("average reward: {:.2f}".format(reward_total / LEARNING_COUNT))
        print("Q Value       :{}".format(agent.q_val))
        gamma_arr[j] = gamma
        list_scores[j] = reward_total / LEARNING_COUNT
        time_array[j] = end-st

        print("###### TEST #####")
        reward_total = 0.0
        for i in range(TEST_COUNT):
            reward = agent.test()
            reward_total += reward
            if i % 100 == 0:
                print(i)
            print(reward)
        print("episodes      : {}".format(TEST_COUNT))
        print("total reward  : {}".format(reward_total))
        print("average reward: {:.2f}".format(reward_total / TEST_COUNT))

        policy_curr = [np.argmax(agent.q_val[state])
                       for state in range(map_size*map_size)]
        policy_curr = np.array(policy_curr)
        policy.append(policy_curr)
        list_scores_test[j] = reward_total / TEST_COUNT

        plot_policy_map(
            f'Q-Learning Policy Map, Gamma: {GAMMA}',
            policy[j].reshape(map_size, map_size),
            desc, colors_lake(), directions_lake()
        )

    print(gamma_arr)
    print(time_array)
    print(list_scores)
    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Q-Learning, Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/Q_ExecutionTime_{map_size}.png")
    plt.show()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Q-Learning, Reward Analysis')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/Q_AverageRewards_{map_size}.png")
    plt.show()

    plt.plot(gamma_arr, list_scores_test)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Q-Learning, Reward Analysis (Test Cases)')
    plt.grid()
    plt.savefig(f"./images/frozen_lake/Q_AverageRewardsTest_{map_size}.png")
    plt.show()
