import gym

from algorihtms import value_iteration, policy_iteration, Q_learning
from envs import game2048_env
env = game2048_env.Game2048Env(size=4)
Q_learning.learn_Q(env)
