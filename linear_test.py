import gym
import numpy as np
from controller.linear import linear_policy


env = gym.make("gym_rocketlander:rocketlander-v0")
w = np.zeros(7)

done = False
observation= env.reset()
while not done:
    action = linear_policy(w, observation)
    observation_, reward, done, info = env.step(action)
    observation = observation_
    env.render()