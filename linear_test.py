import gym
import numpy as np
from controller.linear import linear_policy
from matplotlib import pyplot as plt


env = gym.make("gym_rocketlander:rocketlander-v0")

# Gains
px = 0.5
dx = 1
py = 0.5
dy = 1
pth = 1
dth = 1
pgim = 5
ffw = 0.8
w = [px,dx,py,dy,pth,dth,pgim,ffw]
w = [0.353, 0.783, 0.454, 0.729, 0.946, 0.969, 4.884, 0.666]

x, y, th, vx, vy, vth, Fdes, thdes, taudes= [], [], [], [], [], [], [], [], []
def log(state):
    x.append(state[0])
    y.append(state[1])
    th.append(state[2])
    vx.append(state[7])
    vy.append(state[8])
    vth.append(state[9])

done = False
observation= env.reset()
score = 0
while not done:
    action, f, t, tau = linear_policy(w, observation)
    Fdes.append(f)
    thdes.append(t)
    taudes.append(tau)
    log(observation)
    observation_, reward, done, info = env.step(action)
    observation = observation_
    score += reward
    env.render()
print(score)
plt.figure(1)
plt.plot(th, label = 'th')
plt.plot(thdes, label = 'thdes')
plt.plot(vx, label = 'w')
plt.plot(taudes, label='taudes')
plt.legend()

plt.figure(2)
plt.plot(Fdes)

plt.figure(3)
plt.plot(x,y)

plt.show()