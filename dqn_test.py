import gym
import numpy as np
from controller.agent import Agent


env = gym.make("gym_rocketlander:rocketlander-v0")
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[10], lr=1e-4)
agent.load("trained_model")

done = False
observation= env.reset()
while not done:
    action = agent.choose_action(observation)
    observation_, reward, done, info = env.step(action)
    observation = observation_
    env.render()




