import gym
import numpy as np
from scipy.optimize import minimize

env = gym.make("gym_rocketlander:rocketlander-v0")
  
def eval_rollout(w):
    n_rollouts = 10
    ave_score = 0
    for _ in range(n_rollouts):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = linear_policy(th, observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_
        ave_score += (1/n_rollouts) * score
    print('average score %.2f' % ave_score, 'w: ' % w)
    return -ave_score

w0 = np.ones(7)
wopt = minimize(eval_rollout, w0, method = 'Nelder-Mead')
