import torch
import argparse
import os
from src.environment import create_env
from src.agent import PPOAgent
from src.test import run_test_instance
from src.train import run_training_instance
from config import ENV_NAME, RENDER_MODE, CHECKPOINT_PATH, NUM_EPISODES, NUM_TEST_EPISODES, DEVICE


def main():
    # Argument parser setup for mode selection
    parser = argparse.ArgumentParser(description="Choose mode to run: train or test")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Specify 'train' or 'test' mode")
    args = parser.parse_args()

    # Initialize environment and agent
    start_episode = 1
    checkpoint_file = os.path.join(CHECKPOINT_PATH, "latest_checkpoint.pth")
    env = create_env(map=ENV_NAME, output_path=f"output/{ENV_NAME}.mp4")
    in_dim = env.observation_space.shape
    num_actions = env.action_space.n
    agent = PPOAgent(in_dim, num_actions)

    if args.mode == "train":
        print("Starting training mode...")

        # Load checkpoint if available
        if os.path.exists(checkpoint_file):
            print(f"Checkpoint file found at {checkpoint_file}. Loading...")
            checkpoint = torch.load(checkpoint_file)
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            start_episode = checkpoint.get('episode', 1)
            print(f"Resuming from episode {start_episode}\n"
                  f"{checkpoint_file}")

            # Check if `start_episode` exceeds or matches `NUM_EPISODES`
            if start_episode >= NUM_EPISODES:
                print(f"Episode {start_episode} reached or exceeded NUM_EPISODES ({NUM_EPISODES}). Restarting from episode 1.")
                start_episode = 1  # Reset to start from the beginning

        else:
            print(f"No checkpoint found. Starting from scratch.")

        last_episode, latest_reward, latest_loss = start_episode, 0.0, 0.0  # Defaults if run_instance is not completed

        try:
            # Run the training instance, starting from the last saved episode
            last_episode, latest_reward, latest_loss = run_training_instance(agent, env, num_episodes=NUM_EPISODES, render=RENDER_MODE, start_episode=start_episode)

        except KeyboardInterrupt:
            print("Manual interrupt detected. Saving progress before exit...")

        finally:
            # Final save on completion or interruption
            last_completed_episode = last_episode if 'last_episode' in locals() else start_episode
            agent.save()  # Calls agent.save() which saves to ACTOR_PATH and CRITIC_PATH
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'episode': last_completed_episode,
                'latest_reward': latest_reward,
                'latest_loss': latest_loss
            }, checkpoint_file)
            print(f"Progress saved to {checkpoint_file}")

    elif args.mode == "test":
        print("Starting testing mode...")
        print(f"WARNING! Initializing Black Hole Device start-up sequence.\n"
              f"WARNING! Please stand back. Local micro-anomalies may occur during testing.\n")

        # Load the latest checkpoint
        checkpoint_file = os.path.join(CHECKPOINT_PATH, "latest_checkpoint.pth")
        if os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"Running test with model from episode {checkpoint.get('episode', 1)}")
        else:
            print("No checkpoint found. Please train the agent before testing.")
            return

        # Test the agent
        run_test_instance(agent, env, num_episodes=NUM_TEST_EPISODES)

    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()



