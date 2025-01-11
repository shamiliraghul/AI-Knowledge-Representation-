import torch
import torch.nn as nn
import random
from collections import deque

# Deep Q neural network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Saving and loading the trained model
def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

def load_model(file_name, state_size, action_size):
    model = QNetwork(state_size, action_size)
    model.load_state_dict(torch.load(file_name))
    return model
