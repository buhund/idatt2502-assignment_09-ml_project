# src/train.py
# Run the training sequence

import torch
from src.agent import PPOAgent
from config import ENV_NAME, RENDER_MODE, DEVICE, NUM_EPISODES
from config import ACTOR_PATH, CRITIC_PATH


# Use GPU/Cuda if available. Else fallback to good 'ol CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE


def load_trained_agent(in_dim, num_actions, actor_path, critic_path):
    """Initialize a PPOAgent and load the trained actor and critic model.

    Args:
        in_dim (tuple): Input dimensions (shape) for the CNN.
        num_actions (int): Number of actions in the action space.
        actor_path (str): Path to the saved actor model.
        critic_path (str): Path to the saved critic model.

    Returns:
        PPOAgent: The PPO agent with loaded weights on the appropriate device (GPU if available).

    """
    agent = PPOAgent(in_dim, num_actions)

    # Load the trained model weights to the device
    agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    agent.critic.load_state_dict(torch.load(critic_path, map_location=device))

    # Move models to the device
    agent.actor.to(device)
    agent.critic.to(device)

    agent.actor.eval()
    agent.critic.eval()

    return agent


def run_instance(agent, env, num_episodes=NUM_EPISODES, render=RENDER_MODE):
    """Plays multiple games with the specified agent and environment.

    Args:
        agent (PPOAgent): The trained agent used to play the game.
        env (gym.Env): The environment in which the game is played.
        num_episodes (int): Number of episodes to run.
        render (bool): If True, renders the environment during gameplay.
    """
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            action, _, _ = agent.select_action(state_tensor)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

            state = next_state

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon}")

    env.close()
