import numpy as np
import tensorflow as tf
import gym
from collections import deque
import random

# 使用PyTorch的DQN模型
import torch
import torch.nn as nn
import torch.optim as optim

class UAVNetworkEnv(gym.Env):
    def __init__(self):
        self.grid_size = 1000  # 3D网格的大小
        self.max_fly_height = 300  # 最大飞行高度
        self.num_uavs = 5  # 无人机数量
        self.action_space = 6  # 动作空间，例如：选择链路、调整飞行路径等
        self.observation_space = 4  # 状态空间：飞行位置、信号强度、链路质量等
        self.state = np.zeros((self.num_uavs, 3))  # 每个无人机的初始位置 [x, y, z]

    def reset(self):
        self.state = np.random.rand(self.num_uavs, 3) * np.array([self.grid_size, self.grid_size, self.max_fly_height])
        return self.state

    def step(self, action):
        # 确保action是一个整数，而不是列表
        if isinstance(action, int):
            for i in range(self.num_uavs):
                if action == 0:
                    self.state[i, 0] += 1  # 向前移动
                elif action == 1:
                    self.state[i, 0] -= 1  # 向后移动
                elif action == 2:
                    self.state[i, 1] += 1  # 向左移动
                elif action == 3:
                    self.state[i, 1] -= 1  # 向右移动
                elif action == 4:
                    self.state[i, 2] += 1  # 向上移动
                elif action == 5:
                    self.state[i, 2] -= 1  # 向下移动

        next_state = self.state
        reward = self.calculate_reward(next_state, action)
        done = False  # 如果满足某些条件则结束，比如所有无人机完成任务
        return next_state, reward, done, {}

    def calculate_reward(self, state, action):
        reward = 0
        for i in range(self.num_uavs):
            reward += np.sum(state[i, :])  # 简单示例：奖励与位置相关
        return reward
    
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 返回一个整数动作
        state = torch.tensor(state, dtype=torch.float32)
        return torch.argmax(self.model(state)).item()  # 选择Q值最大的动作

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float32)
                target += self.gamma * torch.max(self.model(next_state)).item()
            state = torch.tensor(state, dtype=torch.float32)
            output = self.model(state)[action]
            loss = nn.MSELoss()(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = UAVNetworkEnv()
    agent = DQNAgent(state_size=3, action_size=6)
    episodes = 1000
    batch_size = 32
    for e in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f"Episode {e+1}/{episodes} completed.")