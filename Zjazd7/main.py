import gym  
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from rl.agents import DQNAgent  
from rl.policy import BoltzmannQPolicy  
from rl.memory import SequentialMemory

"""
This script demonstrates training a Deep Q-Network (DQN) agent using the Gym library on the CartPole-v1 environment.

Authors: Dariusz Karasiewicz, Mikołaj Kusiński
"""

def create_dqn_agent(model, actions):
    """
    Create a Deep Q-Learning agent for training on the CartPole-v1 environment.

    Parameters:
        - model (Sequential): DQN model for the CartPole-v1 environment.
        - actions (int): Number of possible actions in the environment.

    Returns:
        - agent (DQNAgent): DQN agent configured for the CartPole-v1 environment.
    """
    agent = DQNAgent(
        model=model,
        memory=SequentialMemory(limit=50000, window_length=1),
        policy=BoltzmannQPolicy(),
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=0.01
    )
    return agent

def create_cartpole_model(states, actions):
    """
    Create a Deep Q-Network (DQN) model for the CartPole-v1 environment.

    Parameters:
        - states (int): Number of observation states in the environment.
        - actions (int): Number of possible actions in the environment.

    Returns:
        - model (Sequential): DQN model for the CartPole-v1 environment.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model

def compile_and_train_agent(agent, env, nb_steps=100000):
    """
    Compile and train the Deep Q-Learning agent on the CartPole-v1 environment.

    Parameters:
        - agent (DQNAgent): DQN agent configured for the CartPole-v1 environment.
        - env: Gym environment.
        - nb_steps (int): Number of training steps.

    Returns:
        None
    """
    agent.compile(tf.keras.optimizers.legacy.Adam(learning_rate=0.001), metrics=["mae"])
    agent.fit(env, nb_steps=nb_steps, visualize=True, verbose=1)

def evaluate_agent(agent, env, nb_episodes=10):
    """
    Evaluate the performance of the Deep Q-Learning agent on the CartPole-v1 environment.

    Parameters:
        - agent (DQNAgent): DQN agent configured for the CartPole-v1 environment.
        - env: Gym environment.
        - nb_episodes (int): Number of evaluation episodes.

    Returns:
        results (dict): Evaluation results.
    """
    results = agent.test(env, nb_episodes=nb_episodes, visualize=True)
    return results

if __name__ == "__main__":
    """
      Create CartPole environment
    """
    env = gym.make("CartPole-v1")
    """
        Define states and actions
    """
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    model = create_cartpole_model(states, actions)
    print(model.summary())

    agent = create_dqn_agent(model, actions)
    print(agent.get_config())

    compile_and_train_agent(agent, env)
    
    evaluation_results = evaluate_agent(agent, env)
    print(f"Average episode reward: {np.mean(evaluation_results.history['episode_reward'])}")

    env.close()