# src/train.py
# Run the training sequence

import torch
from src.agent import PPOAgent
from config import ENV_NAME, RENDER_MODE, DEVICE, NUM_EPISODES, CHECKPOINT_INTERVAL, CHECKPOINT_PATH
from config import ACTOR_PATH, CRITIC_PATH
from torch.utils.tensorboard import SummaryWriter


def load_trained_agent(in_dim, num_actions, actor_path, critic_path):
    """Initialize a PPOAgent and load the trained actor and critic model weights."""
    agent = PPOAgent(in_dim, num_actions)

    # Load the trained model weights to the device
    agent.actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    agent.critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))

    # Move models to the device
    agent.actor.to(DEVICE)
    agent.critic.to(DEVICE)

    agent.actor.eval()
    agent.critic.eval()

    return agent


def run_instance(agent, env, num_episodes=NUM_EPISODES, render=RENDER_MODE, start_episode=1):
    """Run the emulator with the specified agent and environment."""
    for episode in range(start_episode - 1, num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            agent.save_checkpoint(CHECKPOINT_PATH, episode + 1)

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            action, _, _ = agent.select_action(state_tensor)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if render:
                env.render()

            state = next_state

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")

    # env.close() # Closing env only in main.py now
    return episode + 1  # Return the last completed episode




