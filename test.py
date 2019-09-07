import gym
from RL_brain import DeepQNetwork
import numpy as np

env = gym.make('CarRacing-v0')
env = env.unwrapped

# print(env.action_space)
# print(env.observation_space)

# observation = env.reset()

# print observation

s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
action = [0.5, 0.5, 0.5]
s_ = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]

transition = [s, action, 2, s_]

# print transition
observation = []

for i in range(100):
    observation.append(transition)

sample = np.array([observation[i] for i in np.random.choice(100, 24)])

# print sample[:,1]

ac_map = [[-1,1,0], [-0.5,1,0], [0,1,0], [0.5,1,0], [1,1,0],
            [-1,1,0.5], [-0.5,1,0.5], [0,1,0.5], [0.5,1,0.5], [1,1,0.5],
            [-1,0,1], [-0.5,0,1], [0,0,1], [0.5,0,1], [1,0,1]]

# env.reset()
# while True:
#     env.render()
#     print env.step(ac_map[np.random.choice(15,1)[0]])
