import gym
import numpy as np
from controller.linear import linear_policy
from scipy.optimize import minimize

env = gym.make("gym_rocketlander:rocketlander-v0")
np.set_printoptions(precision=3)

def eval_rollout(w):
    n_rollouts = 100
    ave_score = 0
    for _ in range(n_rollouts):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action, _, _, _ = linear_policy(w, observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_
        ave_score += (1/n_rollouts) * score
    print('average score %.2f' % ave_score, 'w: ' + str(w))
    return -ave_score

px = 0.5
dx = 0.8
py = 0.5
dy = 1
pth = 1
dth = 1
pgim = 5
ffw = 0.8
w0 = [px,dx,py,dy,pth,dth,pgim,ffw]
w0 = [0.353, 0.783, 0.454, 0.729, 0.946, 0.969, 4.884, 0.666]
# print(-eval_rollout(w0))
wopt = minimize(eval_rollout, w0, method='Nelder-Mead')
