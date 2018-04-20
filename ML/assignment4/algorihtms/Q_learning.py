"""
Q-Learning example using OpenAI gym MountainCar enviornment
Author: Moustafa Alzantot (malzantot@ucla.edu)

This class has been modified to approximate Q using a NN
"""
import numpy as np
from sklearn.neural_network import MLPRegressor

import gym
from gym import wrappers

n_states = 40
iter_max = 10000

initial_lr = 1.0  # Learning rate
min_lr = 0.003
gamma = 1.0
t_max = 10000
eps = 0.02


def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a, b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward


def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    return a, b


if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    print('----- using Q Learning -----')
    q_table = MLPRegressor((100, 100), learning_rate='invscaling', learning_rate_init=0.85)
    q_table.fit([[0, 0, 0], [1, 1, 1]], [0, 0])
    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, initial_lr * (0.85 ** (i // 100)))
        for j in range(t_max):
            a, b = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < eps:
                action = np.random.choice(env.action_space.n)
            else:
                logits = [q_table.predict([[a, b, action]])[0] for action in range(env.action_space.n)]
                logits_exp = np.exp(logits)
                if sum(logits_exp) == 0:
                    probs = [1 / env.action_space.n] * env.action_space.n
                else:
                    probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(env.action_space.n, p=probs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            # update q table
            a_, b_ = obs_to_state(env, obs)
            q_table.partial_fit(
                [[a, b, action]],
                [reward + gamma * np.max(
                    [q_table.predict([[a, b, action]])[0] for action in range(env.action_space.n)]) -
                 q_table.predict([[a, b, action]])[0]]
            )
            if done:
                break
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' % (i + 1, total_reward))
    # solution_policy = np.argmax(q_table, axis=2)
    # solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    # print("Average score of solution = ", np.mean(solution_policy_scores))
    # # Animate it
    # run_episode(env, solution_policy, True)
