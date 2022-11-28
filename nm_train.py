import gym
import numpy as np
from controller.linear import linear_policy
from scipy.optimize import minimize

env = gym.make("gym_rocketlander:rocketlander-v0")
np.set_printoptions(precision=3)

def eval_rollout(w):
    n_rollouts = 50
    ave_score = 0
    for _ in range(n_rollouts):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = linear_policy(w, observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_
        ave_score += (1/n_rollouts) * score
    print('average score %.2f' % ave_score, 'w: ' + str(w))
    return -ave_score

w0 = 0.01*np.ones(8)
wopt = minimize(eval_rollout, w0)
