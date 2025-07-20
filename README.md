# Deep_Q_Learning with Atari
This project implements a Deep Q-Network (DQN) agent to play Atari's Breakout game using Stable-Baselines3 and Gymnasium. The implementation includes both training and evaluation scripts.

## Project Structure
dqn-breakout/

│

├── train.py          # Script for training the DQN agent

├── play.py           # Script for evaluating the trained agent

├── README.md         # This file

│

├── models/           # Directory for saved models (created during training)

│   └── policy.zip    # Trained model file (created after training)

│   └── dqn_model_config1_cnn.zip

│   └── dqn_model_config2_mlp.zip

│

└── logs/            # Training logs for tensorboard (created during training)

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

## Add hyperparameter table here

## Video Demonstration

[link here]
The video demonstrates:

- Agent successfully interacting with the environment
- Real-time gameplay visualization
- Trained policy making intelligent decisions
- GUI rendering of the game state

## Contributors

- Abubakar Ahmed 
- Theodora Ngozichukwuka Omunizua
  
