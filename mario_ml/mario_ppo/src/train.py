# src/train.py
# Run the training sequence

import torch
from src.agent import PPOAgent
from config import ENV_NAME, RENDER_MODE, DEVICE, NUM_EPISODES, CHECKPOINT_INTERVAL, CHECKPOINT_PATH
from config import ACTOR_PATH, CRITIC_PATH
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import csv
import os


def load_trained_agent(in_dim, num_actions, actor_path, critic_path):
    """Initialize a PPOAgent and load the trained actor and critic model weights."""


    agent = PPOAgent(in_dim, num_actions)
    # Printout to verify when/if the method is called
    print(f"Loading the trained Actor and Critic model weithts:\n"
          f"Actor: {actor_path}\n"
          f"Critic: {critic_path}\n"
          f"in_dim: {in_dim}\n"
          f"num_actions: {num_actions}\n")
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
    """Run the emulator with the specified agent and environment, logging metrics."""

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/mario_ppo")

    # Initialize CSV file for logging
    csv_path = os.path.join("logs", "training_metrics.csv")
    os.makedirs("logs", exist_ok=True)

    with open(csv_path, mode="w", newline="") as csv_file:
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow(["Episode", "Reward", "Actor_Loss", "Critic_Loss"])  # CSV headers

    for episode in range(start_episode - 1, num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        episode_loss = 0
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        while not done:
            # Get state and choose action
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            action, log_prob, value = agent.select_action(state_tensor)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # Store and store step information data for PPO update
            states.append(state_tensor)
            actions.append(torch.tensor(action, device=agent.device))
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            state = next_state

            if render:
                env.render()

        # Calculate returns and advantages
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            next_value = agent.critic(next_state_tensor).squeeze(-1)

        # Using agent's `compute_gae` method to get advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, next_value, dones)

        # Convert lists to tensors
        states = torch.cat(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)
        returns = returns.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update the agent and retrieve average losses
        avg_actor_loss, avg_critic_loss = agent.update_loss_tracking(states, actions, old_log_probs, returns, advantages)
        episode_loss = avg_actor_loss + avg_critic_loss

        # Log metrics to TensorBoard
        writer.add_scalar("Reward/Episode", total_reward, episode)
        writer.add_scalar("Actor_Loss/Episode", avg_actor_loss, episode)
        writer.add_scalar("Critic_Loss/Episode", avg_critic_loss, episode)

        # Log metrics to CSV
        with open(csv_path, mode="a", newline="") as csv_file:
            writer_csv = csv.writer(csv_file)
            writer_csv.writerow([episode + 1, total_reward, avg_actor_loss, avg_critic_loss])

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}, "
              f"Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")

    writer.close()  # Close the TensorBoard writer
    return episode + 1, total_reward, episode_loss





