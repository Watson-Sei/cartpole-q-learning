import gymnasium as gym
from .cartpole_q_learning import train, inference

def hello():
    print('Hello from cartpole!')

def train_start():
    env = gym.make('CartPole-v1')
    q_network = train(env)

    env.close()

def inference_start():
    env = gym.make('CartPole-v1', render_mode='human')

    inference(env, 'q_network.pth')

    env.close()