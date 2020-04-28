import math, random

import cv2
import gym
import numpy as np
from collections import deque


import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt
#%matplotlib inline


def test(env_id, model):
    """
    Method to play 30 games with trained agent.

    Parameters
    ----------
    env_id: str
        name of your environment
    model: model

    Returns
    all_test_rewards: list
        list of rewards collected in 30 episodes.
    """
    env = gym.make(env_id)
    all_test_rewards = list()
    episode_reward = 0
    episode_count = 0
    state = env.reset()
    plt.figure(figsize=(9, 9))
    img = plt.imshow(env.render(mode='rgb_array'))  # only call this once
    while True:
        if episode_count == 30:
            break
        img.set_data(env.render(mode='rgb_array'))  # just update the data
        display.display(plt.gcf())
        display.clear_output(wait=True)
        action = model.act(state, -1)  # -ve epsilon so it always choses the policy it learned rather than random action
        next_state, reward, done, _ = env.step(action)
        if not done:
            episode_reward += reward
        else:
            state = env.reset()
            all_test_rewards.append(episode_reward)
            print(episode_reward)
            episode_reward = 0
            episode_count += 1
    print("The mean score of your agent: ", np.mean(all_test_rewards))
    env.close()
    return all_test_rewards

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

env_id = "CartPole-v0"
env = gym.make(env_id)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

plt.plot([epsilon_by_frame(i) for i in range(10000)])


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state).cpu()
            action = q_value.max(1)[1].data[0].numpy().tolist()
        else:
            action = random.randrange(self.num_actions)
        return action


model_batch = DQN(env.observation_space.shape[0], env.action_space.n)

if USE_CUDA:
    model_batch = model_batch.cuda()

optimizer = optim.Adam(model_batch.parameters())


class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Parameters
        ----------
        capacity: int
            the length of your buffer
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        batch_size: int
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def compute_td_loss_batch(batch_size):
    """
    Parameters
    ----------
    batch_size: int

    Returns
    -------
    loss: tensor
    """
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    with torch.no_grad():
        next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    # TODO: predict q_values.
    # Hint:- remeber it's a batch prediction so there will be extra dimention.
    q_values = model_batch(state)

    # TODO: predict next state's q_values.
    next_q_values = model_batch(next_state)

    # TODO: get the q_values based on actions you took.
    # Hint:- the logic should be same as previous one but remember there is an extra dimention.
    action = action.unsqueeze(1)
    q_values = q_values#.squeeze(1)
    q_value = q_values.gather(1, action)

    next_q_value = next_q_values.max(1)[0]

    # TODO: calculate expected q value based on bellman eq.
    expected_q_value = reward + gamma * next_q_value

    # TODO: calculate MSE
    loss = F.mse_loss(expected_q_value, q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def plot(frame_idx, rewards, losses):
    """
    Parameters
    ----------
    frame_idx: int
        frame id
    rewards: int
        accumulated reward
    losses: int
        loss
    """
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    num_frames = 20000
    batch_size = 32
    gamma = 0.99
    replay_buffer = ReplayBuffer(1000)
    losses = []
    all_rewards = []
    episode_reward = 0
    count = 0
    state = env.reset()
    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = model_batch.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss_batch(batch_size)
            loss = loss.data.numpy().tolist()
            losses.append(loss)

        if frame_idx % 200 == 0:
            count = plot(frame_idx, all_rewards, losses)