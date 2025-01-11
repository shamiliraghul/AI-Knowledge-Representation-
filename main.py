import pandas as pd
import matplotlib.pyplot as plt
from policy import tabular_q_policy, train_dqn
from model import save_model

# data loading and training
#1.Data Pre_Processing
def load_and_preprocess_data(data):
    # checking all relevant columns are numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Dropping null values
    data.dropna(inplace=True)

    # Computing state function:RSI
    def compute_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data['RSI'] = compute_rsi(data)
    data.dropna(inplace=True)

    # Normalizing Features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI']] = scaler.fit_transform(
        data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI']]
    )

    return data

if __name__ == "__main__":
    # Loading multi_stock datasets from the source yfinance(Jan 2008 - Dec 2022)
    stock1 = pd.read_csv('aapl.csv')
    stock2 = pd.read_csv('googl.csv')
    stock3 = pd.read_csv('msft.csv')

    data1 = load_and_preprocess_data(stock1)
    data2 = load_and_preprocess_data(stock2)
    data3 = load_and_preprocess_data(stock3)
    #State Function
    state_features = ['Open', 'High', 'Low', 'Close', 'RSI']
    #Action(Decision)
    action_space = ['Buy', 'Sell', 'Hold']

    # Training the model for each stock using Tabular Q-Learning algo
    tabular_rewards = []
    for i, data in enumerate([data1, data2, data3], start=1):
        print(f"Training Tabular Q-Policy for Stock {i}")
        rewards = tabular_q_policy(data, state_features)
        tabular_rewards.append(rewards)

    # Training the model for each stock using Deep Q Neural Network algo
    dqn_rewards = []
    dqn_models = []
    for i, data in enumerate([data1, data2, data3], start=1):
        print(f"Training DQN for Stock {i}")
        model, rewards = train_dqn(data, state_features, action_space)
        dqn_rewards.append(rewards)
        dqn_models.append(model)
        save_model(model, f"stock_{i}_dqn.pth")

    # Cummulative Reward Visualization- Tabular Q Learning
    plt.figure(figsize=(12, 8))
    for i, rewards in enumerate(tabular_rewards, start=1):
        plt.plot(rewards, label=f'Stock {i} Tabular Q-Policy')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Tabular Q-Policy Rewards for Multiple Stocks')
    plt.legend()
    plt.show()

    # Cummulative Reward Visualization-DQN Algo
    plt.figure(figsize=(12, 8))
    for i, rewards in enumerate(dqn_rewards, start=1):
        plt.plot(rewards, label=f'Stock {i} DQN Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Rewards for Multiple Stocks')
    plt.legend()
    plt.show()

    # Observation: Cummulative Reward comparison for each stock
    for i, (tab_rewards, dqn_rewards) in enumerate(zip(tabular_rewards, dqn_rewards), start=1):
        plt.figure(figsize=(6, 4))
        total_tabular_reward = sum(tab_rewards)
        total_dqn_reward = sum(dqn_rewards)
        plt.bar(['Tabular Q', 'DQN'], [total_tabular_reward, total_dqn_reward], color=['blue', 'orange'])
        plt.title(f'Comparison of Policies for Stock {i}')
        plt.ylabel('Total Reward')
        plt.show()
