import gym
import numpy as np
from controller.agent import Agent

env = gym.make("gym_rocketlander:rocketlander-v0")
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[10], lr=0.001)
scores, eps_history = [], []
n_games = 1000

for i in range(n_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation, train=True)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_
    scores.append(score)
    eps_history.append(agent.epsilon)

    ave_score = np.mean(scores[-100:])

    print('episode ', i, 'score %.2f' % score, 'average score %.2f' % ave_score, 'epsilon %.2f' % agent.epsilon)

path = "./trained_model"
agent.save(path)
