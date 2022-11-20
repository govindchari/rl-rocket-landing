import gym

env = gym.make("gym_rocketlander:rocketlander-v0")

done = False
observation = env.reset()
while not done:
    action = env.sample_action(observation)
    observation_, reward, done, _, info = env.step(action)
    observation = observation_
    env.render()