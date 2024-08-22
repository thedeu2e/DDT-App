import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 5

NUM_ACTIONS = 10
NUM_STATES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    def __init__(self, epsilon_decay=0.995, min_epsilon=0.01):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).to(device), 0)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()  # move back to CPU
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES]).to(device)
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int)).to(device)
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2]).to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:]).to(device)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, file_path):
        torch.save({
            'eval_net_state_dict': self.eval_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'memory': self.memory,
            'memory_counter': self.memory_counter,
            'learn_step_counter': self.learn_step_counter
        }, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.eval_net.load_state_dict(checkpoint['eval_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.min_epsilon = checkpoint['min_epsilon']
        self.memory = checkpoint['memory']
        self.memory_counter = checkpoint['memory_counter']
        self.learn_step_counter = checkpoint['learn_step_counter']
