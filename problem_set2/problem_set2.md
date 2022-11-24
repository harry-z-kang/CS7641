# Problem Set 2 -- Zixuan Kang --903404772

## 1. You have to communicate a signal in a language that has 3 symbols A, B and C. The probability of observing A is 50% while that of observing B and C is 25% each. Design an appropriate encoding for this language. What is the entropy of this signal in bits?

## 2. Show that the K-means procedure can be viewed as a special case of the EM algorithm applied to an appropriate mixture of Gaussian densities model.

## 3. Plot the direction of the first and second PCA components in the figures given.

![Q3 Image](./q3.png)

## 4. Which clustering method(s) is most likely to produce the following results at k = 2? Choose the most likely method(s) and briefly explain why it/they will work better where others will not in at most 3 sentences. Here are the five clustering methods you can choose from:

- Hierarchical clustering with single link
- Hierarchical clustering with complete link
- Hierarchical clustering with average link
- K-means
- EM

### a. ![Q4 A Image](./q4a.png)

### b. ![Q4 B Image](./q4b.png)

### c. ![Q4 C Image](./q4c.png)

## 5. You receive the following letter

> Dear Friend,
> 
> Some time ago, I bought this old house, but found it to be haunted by ghostly sardonic laughter.
> As a result it is hardly habitable. There is hope, however, for by actual testing I have found that
> this haunting is subject to certain laws, obscure but infallible, and that the laughter can be
> affected by my playing the organ or burning incense.
> 
> In each minute, the laughter occurs or not, it shows no degree. What it will do during the
> ensuing minute depends, in the following exact way, on what has been happening during the
> preceding minute:
> 
> Whenever there is laughter, it will continue in the succeeding minute unless I play the organ, in
> which case it will stop. But continuing to play the organ does not keep the house quiet. I notice,
> however, that whenever I burn incense when the house is quiet and do not play the organ it
> remains quiet for the next minute.
> 
> At this minute of writing, the laughter is going on. Please tell me what manipulations of incense
> and organ I should make to get that house quiet, and to keep it so.
> 
> Sincerely,
> At Wit's End

### a. Formulate this problem as an MDP. (For the sake of uniformity, formulate it as a continuing discounted problem, with gamma = 0.9. Let the reward be +1 on any transition into the silent state, and -1 on any transition into the laughing state.) Explicitly give the state set, action sets, state transition, and reward function.

### b. Start with policy pi(laughing) = pi(silent) = (incense, no organ). Perform a couple of steps of policy iteration (by hand!) until you find an optimal policy. (Clearly show and label each step. If you are taking a lot of iteration, stop and reconsider your formulation!)

### c. Do a couple of steps of value iteration as well.

### d. What are the resulting optimal state-action values for all state-action pairs?

### e. What is your advice to "At Wit's End"?

## 6. Use the Bellman equation to calculate Q(s, a1) and Q(s, a2) for the scenario shown in the figure. Consider two different policies:

- Total exploration: All actions are chosen with equal probability.
- Greedy exploitation: The agent always chooses the best action.

Note that the rewards/next states are stochastic for the actions a1’, a2’ and a3’. Assume that the probabilities for the outcome of these actions are all equal. Assume that reward gathering / decision making stops at the empty circles at the bottom.

![Q6 Image](./q6.png)

## 7. ​Consider the following simple grid-world problem. (Actions are N, S, E, W and are deterministic.) Our goal is to maximize the following reward:

- 10 for the transition from state 6 to G
- 10 for the transition from state 8 to G
- 0 for all other transitions

| S   | 2   | 3   |
|:---:|:---:|:---:|
| 4   | 5   | 6   |
| 7   | 8   | G   |

### a. Draw the Markov Decision Process associated to the system.

### b. Compute the value function for each state for iteration 0, 1, 2 and 3 with $\gamma=0.8$

## 8. Find a Nash Equilibrium in each case. The rows denote strategies for Player 1 and columns denote strategies for Player 2.

|     | A    | B    |
|:---:|:----:|:----:|
| A   | 2, 1 | 0, 0 |
| B   | 0, 0 | 1, 2 |

|     | A    | B    |
|:---:|:----:|:----:|
| A   | 2, 1 | 1, 2 |
| B   | 1, 2 | 2, 1 |

|     | L    | R    |
|:---:|:----:|:----:|
| T   | 2, 2 | 0, 0 |
| B   | 0, 0 | 1, 1 |
