import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os
import ale_py

gym.register_envs(ale_py)

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def create_env(render_mode=None):
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
    env = Monitor(env)
    return env

# Hyperparameter configurations
hyperparams_sets = [
    # First try - CNN with REDUCED buffer size
    {
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 10000,  
        "learning_starts": 1000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "name": "config1_cnn"
    },
    # Second try - MLP with even smaller buffer
    {
        "policy": "MlpPolicy", 
        "learning_rate": 2.5e-4,
        "gamma": 0.95,
        "batch_size": 32, 
        "buffer_size": 5000,  
        "learning_starts": 1000,
        "target_update_interval": 500,
        "train_freq": 8,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "name": "config2_mlp"
    },
    # Third try - different learning rate with small buffer
    {
        "policy": "CnnPolicy",
        "learning_rate": 5e-4,
        "gamma": 0.98,
        "batch_size": 32,  
        "buffer_size": 10000,  
        "learning_starts": 5000,  
        "target_update_interval": 2000,
        "train_freq": 2,
        "exploration_fraction": 0.15,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.005,
        "name": "config3_higher_lr"
    },
    # Fourth try - smallest configuration
    {
        "policy": "CnnPolicy",
        "learning_rate": 1e-5,
        "gamma": 0.999,
        "batch_size": 16,
        "buffer_size": 5000,  
        "learning_starts": 2000,  
        "target_update_interval": 250,
        "train_freq": 1,
        "exploration_fraction": 0.3,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.1,
        "name": "config4_small_batch"
    }
]

# Create and wrap the environment
env = DummyVecEnv([lambda: create_env()])

# Create evaluation environment
eval_env = DummyVecEnv([lambda: create_env()])

for params in hyperparams_sets:
    print(f"\nTraining with parameters: {params['name']}")
    print(f"Buffer size: {params['buffer_size']}, Batch size: {params['batch_size']}")
    
    try:
        # Create model
        model = DQN(
            params["policy"],
            env,
            verbose=1,
            tensorboard_log=f"logs/{params['name']}",
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            buffer_size=params["buffer_size"],
            learning_starts=params["learning_starts"],
            target_update_interval=params["target_update_interval"],
            train_freq=params["train_freq"],
            exploration_fraction=params["exploration_fraction"],
            exploration_initial_eps=params["exploration_initial_eps"],
            exploration_final_eps=params["exploration_final_eps"]
        )
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"models/{params['name']}",
            log_path=f"logs/{params['name']}",
            eval_freq=5000,  
            deterministic=True,
            render=False
        )
        
        # Train the model with REDUCED timesteps for memory efficiency
        model.learn(
            total_timesteps=100000,  
            callback=eval_callback,
            tb_log_name=params["name"]
        )
        
        # Save the final model
        model_path = f"models/dqn_model_{params['name']}.zip"
        model.save(model_path)
        
        print(f"\nTraining completed for {params['name']}")
        
        # Clear model from memory
        del model
        
    except Exception as e:
        print(f"Error training {params['name']}: {e}")
        print("Skipping this configuration due to memory constraints...")
        continue

env.close()
eval_env.close()
print("All training completed!")
