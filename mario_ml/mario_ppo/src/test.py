# src/test.py
"""
This will run the trained Agent on the environment without accumulating further knowledge.
The purpose is to test the trained model's performance in the environment.
"""
from tabnanny import check

import torch
import os
from src.environment import create_env
from src.agent import PPOAgent
from torch.utils.tensorboard import SummaryWriter
from config import ENV_NAME, DEVICE, CHECKPOINT_PATH, NUM_TEST_EPISODES


def run_test_instance(agent, env, num_episodes=NUM_TEST_EPISODES):
    # Funny guy
    print(f"Testing Mode Enabled.\n"
          f"WARNING! Initializing Black Hole Device start-up sequence.\n"
          f"WARNING! Please stand back. Local micro-anomalies may occur during testing.\n")

    writer = SummaryWriter(log_dir="runs/mario_ppo_testing")
    csv_path = os.path.join("logs", "testing_metrics.csv")

    with open(csv_path, mode="w") as csv_file:
        csv_file.write("Episode,Total_Reward,x_pos,Flag_Get,Total_Time\n")  # CSV headers

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            start_time = 0
            x_pos = 0
            flag_get = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                action, _, _ = agent.get_action(state_tensor)
                next_state, reward, done, info = env.step(action)
                total_reward += reward

                # Capture initial time, x_pos and flag status from info
                if start_time is None:
                    start_time = info['time']
                x_pos = info.get('x_pos', x_pos) # Update x_pos
                flag_get = info.get('flag_get', flag_get) # Update flag status if reached

                state = next_state

            # Calculate the in-game time spent based on the remaining time in `info`
            # Time counts downward from the start time.
            end_time = info['time']
            total_time = start_time - end_time

            # Write to CSV
            csv_file.write(f"{episode + 1},{total_reward},{x_pos},{int(flag_get)},{total_time}\n")

            # Log to Tensorboard
            writer.add_scalar("Testing/Total Reward", total_reward, episode + 1)
            writer.add_scalar("Testing/x_pos", x_pos, episode + 1)
            writer.add_scalar("Testing/Flag_Get", flag_get, episode + 1)
            writer.add_scalar("Testing/Total Time", total_time, episode + 1)

            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}, "
                  f"x_pos: {x_pos}, Flag_Get: {flag_get}, Total Time: {total_time}")

    writer.close()
    env.close()
    print(f"Testing complete and environment closed.")


def main():
    # Loads the latest checkpoint available for the most up-to-date model
    checkpoint_file = os.path.join(CHECKPOINT_PATH, "latest_checkpoint.pth")

    # Initialize the environemtn
    env = create_env(map=ENV_NAME)
    in_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # Initialize the agent
    agent = PPOAgent(in_dim, num_actions)

    # Load the trained model if checkpoint exists
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}. Loading...")
        checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Training the model from episode {checkpoint.get('episode', 1)}")
    else:
        print(f"No checkpoint found at {checkpoint_file}\n"
              f"Please train the agent before running in Testing Mode")
        return

    run_test_instance(agent, env, num_episodes=NUM_TEST_EPISODES)

if __name__ == '__main__':
    main()











