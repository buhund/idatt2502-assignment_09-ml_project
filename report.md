# IDATT2502 Machine Learning Project



**Project: Diverse Reinforcement Learning oppgaver**

*Velg et environment (for eksempel fra [Gymnasium](https://gymnasium.farama.org/)) - test forskjellige algoritmer*


## Algorithms to explore:

Algorithms that can be effective for this type of environment:

- Deep Q-Network (DQN): Suitable for environments with discrete action spaces like Super Mario Bros. A DQN is a type of neural network that approximates the Q-value function, allowing the agent to decide the best action based on the current game state.
- Double DQN: This variation addresses the overestimation bias in standard DQNs, providing better performance in some scenarios. It helps by stabilizing learning, which can be critical in the dynamic settings of games. 
- Policy Gradient Methods (e.g., REINFORCE): These algorithms directly optimize the policy and are particularly useful if the environment is highly stochastic or if you want to handle continuous action spaces.
- Proximal Policy Optimization (PPO): Popular in game AI, PPO is a stable and robust policy-gradient method, often preferred over other algorithms due to its efficient learning and ability to handle complex environments. 
- Evolution Strategies (ES): These are population-based methods that can work without backpropagation, providing a scalable alternative to reinforcement learning. ES has been applied in gaming scenarios, though it’s often used as a complement to RL rather than a replacement.