"""
Solving FrozenLake8x8 environment using Value-Itertion.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
import time
from gym import wrappers


def run_episode(env, policy, gamma=1.0, render=False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    max_i = 1000
    while step_idx < max_i:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma=1.0, n=100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
        run_episode(env, policy, gamma=gamma, render=False)
        for _ in range(n)]
    return scores


def extract_policy(v, env, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma=1.0, step=None, max_iterations=100000, eps=1e-20):
    """ Value-iteration algorithm """
    v = np.zeros(env.nS)  # initialize value-function
    all_v = []
    all_t = []
    all_eps = []
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            v[s] = max(q_sa)
        increment = np.sum(np.fabs(prev_v - v))
        if increment <= eps:
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
        if step is not None:
            if divmod(i, step)[1] == 0:
                # print("step:"+str(i)+", increment:"+str(increment))
                all_v.append(np.copy(v))
                all_t.append(time.time())
                all_eps.append(increment)
    if step is not None:
        return {
            "values": all_v,
            "computation_time": all_t,
            "increment": all_eps
        }
    return v

if __name__ == '__main__':
    env_name = 'Taxi-v2'
    gamma = 1.0
    env = gym.make(env_name)
    optimal_v = value_iteration(env.env, gamma)
    policy = extract_policy(optimal_v, env.env, gamma)
    policy_score = evaluate_policy(env.env, policy, gamma, n=1000)
    print('Policy average score = ', policy_score)
