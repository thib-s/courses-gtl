"""
Q-Learning example using OpenAI gym MountainCar enviornment
Author: Moustafa Alzantot (malzantot@ucla.edu)

This class has been modified by Thibaut Boissin (tboissin3@gatech.edu to approximate Q using a NN
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle
import gym
from gym import wrappers

n_states = 1000000
iter_max = 4000

initial_lr = 0.2  # Learning rate
min_lr = 0.003
gamma = 0.5
t_max = 1000
eps = 0.02


# def run_episode(env, policy=None, render=False):
#     obs = env.reset()
#     total_reward = 0
#     step_idx = 0
#     for _ in range(t_max):
#         if render:
#             env.render()
#         if policy is None:
#             action = env.action_space.sample()
#         else:
#             state = obs_to_state(env, obs)
#             action = policy[a][b]
#         obs, reward, done, _ = env.step(action)
#         total_reward += gamma ** step_idx * reward
#         step_idx += 1
#         if done:
#             break
#     return total_reward


def obs_to_state(env, obs):
    """ Maps an observation to state """

    return np.reshape(env.Matrix, (16))


# def obs_to_state(env, obs):
#     """ Maps an observation to state """
#     env_low = env.observation_space.low
#     env_high = env.observation_space.high
#     env_dx = (env_high - env_low) / n_states
#     a = int((obs[0] - env_low[0]) / env_dx[0])
#     b = int((obs[1] - env_low[1]) / env_dx[1])
#     return state


def learn_Q(env):
    rewards = []
    env.seed(0)
    np.random.seed(0)
    print('----- using Q Learning -----')
    obs = env.reset()
    try:
        q_table = pickle.load(open("Q_nn.pkl", 'rb'))
        print("model loaded from disk")
    except:
        print("unable to load model, creating new one")
        q_table = MLPRegressor((100, 100), learning_rate='constant', learning_rate_init=initial_lr)
        q_table.fit([np.concatenate((obs_to_state(env, obs), [0]))], [0])
    for i in range(iter_max):
        env.seed(i)
        np.random.seed(i)
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        # q_table.learning_rate = max(min_lr, initial_lr * (0.85 ** (i // 100)))
        for j in range(t_max):
            state = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < eps:
                action = np.random.choice(env.action_space.n)
            else:
                logits = [q_table.predict([np.concatenate((state, [action]))])[0] for action in range(env.action_space.n)]
                logits_exp = np.exp(logits)
                if sum(logits_exp) == 0:
                    action = np.random.choice(env.action_space.n)
                else:
                    probs = logits_exp / np.sum(logits_exp)
                    action = np.random.choice(env.action_space.n, p=probs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            # update q table
            state_ = obs_to_state(env, obs)
            q_table.partial_fit(
                [np.concatenate((state, [action]))],
                [reward + gamma * np.max(
                    [q_table.predict([np.concatenate((state, [action]))])[0] for action in range(env.action_space.n)]) -
                 q_table.predict([np.concatenate((state_, [action]))])[0]]
            )
            if done:
                break
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' % (i + 1, total_reward))
            rewards.append(total_reward)
            pickle.dump(q_table, open("Q_nn.pkl", 'wb'))
    pickle.dump(rewards, open("Q_rewards.pkl", 'wb'))
