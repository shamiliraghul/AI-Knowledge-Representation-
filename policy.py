import numpy as np
import torch
import pandas as pd
from model import QNetwork, ReplayBuffer

# Policy 1: Tabular Q learning Policy
def tabular_q_policy(data, state_features,
                     episodes=20,
                     alpha=0.1,
                     gamma=0.99,
                     epsilon=1.0,
                     epsilon_min=0.1,
                     epsilon_decay=0.995):
    num_bins = 10  # Discretize state features into bins
    action_space = ['Buy', 'Sell', 'Hold']

    def discretize_state(state):
        binned_state = tuple(pd.cut(state, bins=num_bins, labels=False))
        return binned_state

    state_bins = [num_bins] * len(state_features)
    q_table = np.zeros((*state_bins, len(action_space)))

    rewards_log = []

    for episode in range(episodes):
        total_reward = 0
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        for i in range(len(data) - 1):
            state = data.iloc[i][state_features].values
            next_state = data.iloc[i + 1][state_features].values
            current_price = data.iloc[i]['Close']
            next_price = data.iloc[i + 1]['Close']

            discrete_state = discretize_state(state)
            discrete_next_state = discretize_state(next_state)

            if np.random.rand() < epsilon:
                action = np.random.randint(len(action_space))
            else:
                action = np.argmax(q_table[discrete_state])

            #Reward Function Calculation:
            reward = (next_price - current_price) if action == 0 else (current_price - next_price if action == 1 else 0)

            best_next_action = np.argmax(q_table[discrete_next_state])
            td_target = reward + gamma * q_table[discrete_next_state][best_next_action]
            q_table[discrete_state][action] += alpha * (td_target - q_table[discrete_state][action])

            total_reward += reward

        rewards_log.append(total_reward)
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")

    return rewards_log

# Policy 2:Deep Q Neural Network Policy
def train_dqn(data,
              state_features,
              action_space,
              episodes=20,
              gamma=0.99,
              epsilon=1.0,
              epsilon_min=0.1,
              epsilon_decay=0.995,
              learning_rate=0.001,
              batch_size=64,
              buffer_size=10000):
    # Ensure all state_features are numeric
    for feature in state_features:
        if not np.issubdtype(data[feature].dtype, np.number):
            raise TypeError(f"Feature '{feature}' contains non-numeric values. Check preprocessing.")

    state_size = len(state_features)
    action_size = len(action_space)

    q_network = QNetwork(state_size, action_size)
    target_network = QNetwork(state_size, action_size)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    replay_buffer = ReplayBuffer(buffer_size)
    rewards_log = []

    for episode in range(episodes):
        total_reward = 0
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        for i in range(len(data) - 1):
            # Cast state to float32
            state = np.array(data.iloc[i][state_features].values, dtype=np.float32)
            next_state = np.array(data.iloc[i + 1][state_features].values, dtype=np.float32)

            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            current_price = data.iloc[i]['Close']
            next_price = data.iloc[i + 1]['Close']

            if np.random.rand() < epsilon:
                action = np.random.randint(action_size)
            else:
                with torch.no_grad():
                    action = torch.argmax(q_network(state)).item()

            reward = (next_price - current_price) if action == 0 else (current_price - next_price if action == 1 else 0)

            total_reward += reward

            replay_buffer.add((state, action, reward, next_state))

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.stack(next_states)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = target_network(next_states).max(1)[0]
                    target_q_values = rewards + gamma * max_next_q_values

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())

        rewards_log.append(total_reward)
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")

    return q_network, rewards_log

