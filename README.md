# Soft Actor-Critic Route Planning in CARLA

Reinforcement learning project that uses the [Soft Actor-Critic (SAC) algorithm](https://arxiv.org/abs/1801.01290) to train an autonomous vehicle in the [CARLA Simulator](https://carla.org). The objective is to teach the vehicle to navigate intersections and follow map routes based on high-level commands (for example, “turn left at the next intersection”) while using a camera-based state space.

## Model Structure
- CNN used to extract features from image, concatenated with command. Q-network and Policy-networks are trained based on SAC.

## State Space
- Front camera sensor input providing visual observations.
- High-level navigation command using one-hot encoding indicating the intended maneuver or route directive. Randomly and automatically decided.

## Reward Function Components
- Amount of progress (delta) made along the expected route (calculated using waypoints) at each timestep.
- Alignment of the vehicle’s angle with the current route direction.
- Collision avoidance (strong penalties for collisions or unsafe interactions).

The authors of the third-party repositories that make this work possible are acknowledged in the THIRD_PARTY_LICENSES file.
