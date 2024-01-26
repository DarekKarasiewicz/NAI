import gym  # pip install gym
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf

from rl.agents import DQNAgent  
from rl.policy import BoltzmannQPolicy  
from rl.memory import SequentialMemory

"""
This script demonstrates training a Deep Q-Network (DQN) agent using the Gym library on the CartPole-v1 environment.

Authors: Dariusz Karasiewicz, Mikołaj Kusiński
"""

env = gym.make("CartPole-v1")  

"""
Define the Deep Q-Network (DQN) model for the CartPole-v1 environment.

Parameters:
    - states (int): Number of observation states in the environment.
    - actions (int): Number of possible actions in the environment.

Returns:
    - model (Sequential): Deep Q-Network model for the CartPole-v1 environment.
"""

states = env.observation_space.shape[0]
actions = env.action_space.n

print(states)
print(actions)

model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

"""
Create a Deep Q-Learning agent for training on the CartPole-v1 environment.

Parameters:
    - model (Sequential): Deep Q-Network model for the CartPole-v1 environment.
    - actions (int): Number of possible actions in the environment.

Returns:
    - agent (DQNAgent): Deep Q-Learning agent configured for the CartPole-v1 environment.
"""

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

print(env)
print(env.observation_space)

"""
Compile and train the Deep Q-Learning agent on the CartPole-v1 environment and evaluate its performance.

Parameters:
    - agent (DQNAgent): Deep Q-Learning agent configured for the CartPole-v1 environment.
"""

agent.compile(tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=True, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()