# Deep_Q_Learning with Atari
This project implements a Deep Q-Network (DQN) agent to play Atari's Breakout game using Stable-Baselines3 and Gymnasium. The implementation includes both training and evaluation scripts.

## Project Structure
```
Deep_Q_Learning/
├── train.py                     # Script for training the DQN agent  
├── play.py                      # Script for evaluating the trained agent  
├── README.md                    # This file  
├── models/                      # Directory for saved models (created during training)  
│   ├── policy.zip               # Trained model file (created after training)  
│   ├── dqn_model_config1_cnn.zip  
│   └── dqn_model_config2_mlp.zip  
└── logs/                        # Training logs for TensorBoard (created during training)
```

## Prerequisites
System Requirements
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)
- Required Python Packages
- gymnasium[atari]
- stable-baselines3[extra]
- ale-py

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/abubakar-ahmed/Deep_Q_Learning.git
   cd Deep_Q_Learning

2. **Create and activate a virtual environment (recommended):**
   
On Windows

    ```bash
    python -m venv venv
    venv\Scripts\activate
On macOS/Linux

    ```bash
    python3 -m venv venv
    source venv/bin/activate

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the script**
   ```bash
   python play.py

## Files Description

**train.py**:
Defines and trains the DQN agent, 
Implements hyperparameter tuning, 
Saves the trained model as dqn_model.zip, 
Logs training metrics (rewards, episode lengths, exploration rate)

**play.py**:
Loads the trained model, 
Runs evaluation episodes with GreedyQPolicy, 
Visualizes agent performance in real-time, 
Uses env.render() for GUI display

## 📋 Hyperparameter Tuning Table

| Hyperparameter Set | Noted Behavior |
|--------------------|----------------|
| `policy=CnnPolicy`, `lr=1e-4`, `γ=0.99`, `batch=32`, `buffer=10000`, `ε_start=1.0`, `ε_end=0.01`, `exploration_fraction=0.1` | Stable learning, good for visual tasks, balanced exploration. |
| `policy=MlpPolicy`, `lr=2.5e-4`, `γ=0.95`, `batch=32`, `buffer=5000`, `ε_start=1.0`, `ε_end=0.02`, `exploration_fraction=0.2` | Faster initial learning, better for simple states, longer exploration, Stll didnt work well though. |
| `policy=CnnPolicy`, `lr=5e-4`, `γ=0.98`, `batch=32`, `buffer=10000`, `ε_start=1.0`, `ε_end=0.005`, `exploration_fraction=0.15` | Faster learning, delayed start, near-greedy exploitation. |
| `policy=CnnPolicy`, `lr=1e-5`, `γ=0.999`, `batch=16`, `buffer=5000`, `ε_start=1.0`, `ε_end=0.1`, `exploration_fraction=0.3` | Slow learning, prioritizes long-term reward, slow exploration decay. still didnt work well though. |


## Training Results

Based on the TensorBoard logs, the training showed the following progression:

Key Metrics Observed:

- Episode Length: Started around 67 steps, progressively increased to ~94 steps
- Episode Reward: Improved from ~0.5 to ~1.0+ over training
- Exploration Rate: Decreased from ~0.75 to ~0.14 (epsilon decay)
- Training FPS: Maintained around 2000-4000 FPS

## Hyperparameter Analysis:

- Learning Rate: Values between 1e-5 and 5e-4 tested for different learning speeds
- Gamma (Discount Factor): Range from 0.95-0.999 to balance short vs long-term rewards
- Batch Size: Small batches (16-32) used for memory efficiency while maintaining stable updates
- Buffer Size: Significantly reduced (5000-10000) to fit within system memory constraints
- Exploration Strategy: Extended exploration phases (0.1-0.3 fraction) to handle complex visual environment

## Video Demonstration

https://drive.google.com/file/d/1UtD-TOB-jL_fWZBf1KIRZWbj5P_NNEJr/view?usp=sharing 
The video demonstrates:

- Agent successfully interacting with the environment
- Real-time gameplay visualization
- Trained policy making intelligent decisions
- GUI rendering of the game state

## Contributors

- Abubakar Ahmed 
- Theodora Ngozichukwuka Omunizua
  
