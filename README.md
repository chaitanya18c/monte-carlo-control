# MONTE CARLO CONTROL ALGORITHM

## AIM
The aim is to use Monte Carlo Control in a specific environment to learn an optimal policy, estimate state-action values, iteratively improve the policy, and optimize decision-making through a functional reinforcement learning algorithm.

## PROBLEM STATEMENT
Monte Carlo Control is a reinforcement learning method, to figure out the best actions for different situations in an environment. The provided code is meant to do this, but it's currently having issues with variables and functions.

## MONTE CARLO CONTROL ALGORITHM
### Step 1:
Initialize Q-values, state-value function, and the policy.

### Step 2:
Interact with the environment to collect episodes using the current policy.

### Step 3:
For each time step within episodes, calculate returns (cumulative rewards) and update Q-values.

### Step 4:
Update the policy based on the improved Q-values.

### Step 5:
Repeat steps 2-4 for a specified number of episodes or until convergence.

### Step 6:
Return the optimal Q-values, state-value function, and policy.

## MONTE CARLO CONTROL FUNCTION
```
Developed By: CHAITANYA P S
Register Number: 212222230024
```
```python
from tqdm import tqdm
def mc_control (env, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 5500, max_steps = 300, first_visit = True):

    nS, nA = env.observation_space.n, env.action_space.n

    disc = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    def decay_schedule(init_value, min_value, decay_ratio, n):
        return np.maximum(min_value, init_value * (decay_ratio ** np.arange(n)))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    def select_action(state, Q, epsilon):
        return np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(nA)

    for e in tqdm(range(n_episodes), leave=False):
        traj = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=bool)

        for t, (state, action, reward, _, _) in enumerate(traj):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            n_steps = len(traj[t:])
            G = np.sum(disc[:n_steps] * traj[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        Q_track[e] = Q

    V = np.max(Q, axis=1)
    pi = {s: np.argmax(Q[s]) for s in range(nS)}

    return Q, V, pi
```

## OUTPUT:
### Name: CHAITANYA P S
### Register Number: 212222230024

<img src = "https://github.com/user-attachments/assets/91325384-1e4e-404a-8770-8eadb23430b0" width="50%">

## RESULT:

Monte Carlo Control successfully learned an optimal policy for the specified environment.
