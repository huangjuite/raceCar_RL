import gym
from RL_brain import DeepQNetwork
import numpy as np

env = gym.make('CarRacing-v0')
env = env.unwrapped

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
nfeatures = [None]
for i in enumerate(env.observation_space.shape):
    nfeatures.append(i[1])
print(nfeatures)


total_steps = 0

action_map = [[-1,1,0], [-0.5,1,0], [0,1,0], [0.5,1,0], [1,1,0],
            [-1,1,0.5], [-0.5,1,0.5], [0,1,0.5], [0.5,1,0.5], [1,1,0.5],
            [-1,0.5,1], [-0.5,0.5,1], [0,0.5,1], [0.5,0.5,1], [1,0.5,1]]

RL = DeepQNetwork(n_actions=len(action_map),
                  features=nfeatures,
                  learning_rate=0.05, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

for i_episode in range(500):
    observation = env.reset()
    ep_r = 0
    step = 0

    while True:
        if i_episode > 200:
            env.render()

        if step > 50:
            action = RL.choose_action(observation)
        else:
            action = 2

        observation_, reward, done, info = env.step(action_map[action])

        if total_steps > 50:
            RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if ep_r < 0:
            done = True

        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1
        step += 1

RL.plot_cost()
