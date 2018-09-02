#!/home/nuck/.virtualenvs/udacity_drl_py3/bin/python
# coding: utf-8

# # Deep Q-Network (DQN)
# ---
# In this notebook, you will implement a DQN agent with OpenAI Gym's LunarLander-v2 environment.
# 
# ### 1. Import the Necessary Packages

# In[45]:


import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# ### 2. Instantiate the Environment and Agent
# 
# Initialize the environment in the code cell below.

# In[46]:


env = gym.make('Blackjack-v0')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


# Please refer to the instructions in `Deep_Q_Network.ipynb` if you would like to write your own DQN agent.  Otherwise, run the code cell below to load the solution files.

# In[47]:


from dqn_agent import Agent

agent = Agent(state_size=3, action_size=2, seed=0)
# watch an untrained agent
state = env.reset()
for j in range(200):
    action = agent.act(np.array(state))
#     env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break 
        
env.close()


# In[48]:


np.array(state)


# ### 3. Train the Agent with DQN
# 
# Run the code cell below to train the agent from scratch.  You are welcome to amend the supplied values of the parameters in the function, to try to see if you can get better performance!
# 
# Alternatively, you can skip to the next step below (**4. Watch a Smart Agent!**), to load the saved model weights from a pre-trained agent.

# In[49]:


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(np.array(state), eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=0.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()
x = np.arange(len(scores))
y = scores
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('output.png')
# plt.show()
f = open("data.txt", "w")
for a,b in zip(x,y):
    f.write("{} {}\n".format(a,b))

f.close()
# ### 4. Watch a Smart Agent!
# 
# In the next code cell, you will load the trained weights from file to watch a smart agent!

# In[6]:


# load the weights from file
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))


# ### 5. Explore
# 
# In this exercise, you have implemented a DQN agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
# - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task with discrete actions!
# - You may like to implement some improvements such as prioritized experience replay, Double DQN, or Dueling DQN! 
# - Write a blog post explaining the intuition behind the DQN algorithm and demonstrating how to use it to solve an RL environment of your choosing.  

# In[39]:


# len(list(agent.memory.memory))

