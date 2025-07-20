import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

# Load the trained model
try:
    model = DQN.load("models/dqn_model_config1_cnn.zip")
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Make sure training completed successfully.")
    exit()

# Create the Atari environment with rendering enabled
try:
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    print("Environment created successfully!")
except Exception as e:
    print(f"Error creating environment: {e}")
    # Try alternative environment names
    try:
        env = gym.make("Breakout-v5", render_mode="human")
        print("Using Breakout-v5 environment")
    except:
        env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
        print("Using BreakoutNoFrameskip-v4 environment")

# Number of episodes to run
num_episodes = 5

for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    print(f"Episode {episode + 1} started...")

    while not done:
        # Use GreedyQPolicy (deterministic=True means greedy action selection)
        action, _ = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step_count += 1
        
        # Optional: Limit episode length to avoid infinite loops
        if step_count > 10000:
            break

    print(f"Episode {episode + 1} finished with total reward: {total_reward}, steps: {step_count}")

env.close()
print("Evaluation completed!")
