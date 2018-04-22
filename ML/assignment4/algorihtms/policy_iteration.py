"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
import time
from gym import wrappers


def run_episode(env, policy, gamma=1.0, render=False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    max_idx = 1000
    while step_idx < max_idx:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma=1.0, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(v, env, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-10
    all_v = []
    all_t = []
    all_eps = []
    i = 0
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
        i += 1
    return v


def policy_iteration(env, gamma=1.0, step=None, max_iterations=100000, eps=1e-20):
    """ Policy-Iteration algorithm """
    all_policies = []
    all_t = []
    all_eps = []
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, env, gamma)
        if (np.all(policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i + 1))
            break
        policy = new_policy
        if step is not None:
            if divmod(i, step)[1] == 0:
                # print("step:"+str(i)+", increment:"+str(increment))
                all_policies.append(np.copy(policy))
                all_t.append(time.time())
                all_eps.append(np.sum(policy == new_policy))
    if step is not None:
        return {
            "policies": all_policies,
            "computation_time": all_t,
            "error": all_eps
        }
    return policy


if __name__ == '__main__':
    env_name = 'Taxi-v2'
    env = gym.make(env_name)
    optimal_policy = policy_iteration(env.env, gamma=0.9, step=1)
    scores = evaluate_policy(env.env, optimal_policy, gamma=0.9)
    print('Average scores = ', np.mean(scores))
