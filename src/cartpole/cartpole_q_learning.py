import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Q-Networkの定義
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Hyperparameters
learning_rate = 0.001 # 学習率
gamma = 0.99 # 割引率
episodes = 1500 # エピソード数
max_steps = 500 # 1エピソードの最大ステップ数
epsilon = 1.0 # ε-greedy法のε
epsilon_decay = 0.995 # εの減衰率
epsilon_min = 0.01 # εの最小値
    
def train(env):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_network = QNetwork(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # グラフ作成用の変数
    rewards = []
    avg_rewards = []
    std_rewards = []

    global epsilon
    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        episode_reward = 0
        for step in range(max_steps):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_network(state).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)

            # 距離の罰則
            cart_position = next_state[0]
            distance_penalty = abs(cart_position) ** 2
            reward -= distance_penalty

            episode_reward += reward 

            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target = reward + gamma * q_network(next_state).max().item()

            q_value = q_network(state)[action]
            loss = criterion(q_value, torch.tensor([target]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

            if terminated:
                break

        rewards.append(episode_reward) 
        avg_reward = np.mean(rewards[-100:]) 
        std_reward = np.std(rewards[-100:]) 
        avg_rewards.append(avg_reward)  
        std_rewards.append(std_reward)

        print(f"Episode: {episode+1}, Steps: {step+1}, Reward: {episode_reward}, Avg Reward: {avg_reward:.2f} +/- {std_reward:.2f}")  # 変更

    # 学習回数と報酬のグラフ
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.savefig('reward_graph.png')
    plt.close()

    # 直近100エピソードの報酬平均と標準偏差グラフ
    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards)
    plt.fill_between(range(len(avg_rewards)), np.array(avg_rewards) - np.array(std_rewards),
                     np.array(avg_rewards) + np.array(std_rewards), alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Average Reward over Last 100 Episodes')
    plt.savefig('avg_reward_graph.png')
    plt.close()

    # 学習済みのQ-Networkを保存
    torch.save(q_network.state_dict(), 'q_network.pth')

    return q_network

def inference(env, q_network_path):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_network = QNetwork(state_size, action_size)
    q_network.load_state_dict(torch.load(q_network_path))
    q_network.eval()

    state, _ = env.reset()
    state = torch.FloatTensor(state)

    while True:
        env.render()

        with torch.no_grad():
            action = q_network(state).argmax().item()

        next_state, _, terminated, truncated, _ = env.step(action)
        state = torch.FloatTensor(next_state)

        if terminated or truncated:
            break