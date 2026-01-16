# TD3 Autonomous Car Navigation (Assignment 17) 🏎️

This repository contains a PyTorch implementation of the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** algorithm for autonomous car navigation.

## 🌟 Features
- **Continuous Control**: The agent outputs continuous values for **Steering** (`-30°` to `+30°`) and **Velocity** (`0` to `5`), enabling smooth driving behavior compared to discrete DQN.
- **Actor-Critic Architecture**: Uses an Actor network to predict actions and Twin Critic networks to estimate Q-values, reducing overestimation bias.
- **Robust Training**: Implements target policy smoothing and delayed policy updates for stable learning.
- **Interactive UI**: Built with PyQt6, featuring real-time training visualization, sensor data, and hyperparameter monitoring.

## 📂 Project Structure
- `citymap.py`: The main environment and GUI application. Handles physics, rendering, and the training loop.
- `td3.py`: The core Reinforcement Learning implementation (Actor, Critic, TD3 Agent, ReplayBuffer).
- `city_map.png`: The environment map.

## 🛠️ Installation

Ensure you have Python 3.8+ installed. Install the dependencies:

```bash
pip install torch numpy PyQt6
```

## 🚀 Usage

Run the main script to start the application:

```bash
python citymap.py
```

### Controls
1.  **Select Start Position**: Click anywhere on the road map to place the car.
2.  **Add Targets**: Click to add one or multiple targets (Goals).
3.  **Start Training**: Right-click to finish setup, then press **SPACE** (or click START) to begin training.

## 🧠 Technical Details

### State Space (9 Dimensions)
-   7 Ray-cast Sensor readings (Distance/Intensity)
-   Orientation to Target (Normalized)
-   Distance to Target (Normalized)

### Action Space (2 Dimensions)
-   **Steering**: Continuous value `[-1, 1]` mapped to `[-30°, 30°]`.
-   **Velocity**: Continuous value `[-1, 1]` mapped to `[0, 5]` pixels/step.

### Algorithm (TD3)
-   **Twin Critics**: Two Q-networks minimize overestimation of value estimates.
-   **Target Smoothing**: Gaussian noise is added to target actions during training to robustness.
-   **Delayed Updates**: The policy (Actor) is updated less frequently than the Q-function (Critic).

## 📝 Logs & visualization
The application features a built-in dashboard showing:
-   Real-time Reward Plot
-   Current Steps & Episode Count
-   Live Action Values (Steering/Speed)
-   Detailed logs for Crashes and Goals

## 👨‍💻 Credits
Advanced Reinforcement Learning Assignment (ERAV4 Session 17)
Implementation by **Jayant Guru Shrivastava**.
