import torch
import os
from src.environment import create_env
from src.agent import PPOAgent
from src.train import run_instance
from config import ENV_NAME, RENDER_MODE, CHECKPOINT_PATH, NUM_EPISODES


def main():
    start_episode = 1
    checkpoint_file = os.path.join(CHECKPOINT_PATH, "latest_checkpoint.pth")

    # Initialize environment and agent
    env = create_env(map=ENV_NAME, output_path=f"output/{ENV_NAME}.mp4")
    in_dim = env.observation_space.shape
    num_actions = env.action_space.n
    agent = PPOAgent(in_dim, num_actions)

    # Load checkpoint if available
    if os.path.exists(checkpoint_file):
        print(f"Checkpoint file found at {checkpoint_file}. Loading...")
        checkpoint = torch.load(checkpoint_file)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        start_episode = checkpoint.get('episode', 1)
        print(f"Resuming from episode {start_episode}\n"
              f"{checkpoint_file}")
    else:
        print(f"No checkpoint found at {checkpoint_file}\n"
              f"Starting from scratch.")

    try:
        # Run the training instance, starting from the last saved episode
        last_episode = run_instance(agent, env, num_episodes=NUM_EPISODES, render=RENDER_MODE, start_episode=start_episode)
        #run_instance(agent, env, start_episode=start_episode)

    except KeyboardInterrupt:
        print("Manual interrupt detected. Saving progress before exit...")

    finally:
        # Final save on completion or interruption
        last_completed_episode = last_episode if 'last_episode' in locals() else start_episode
        latest_reward = total_reward  # Assuming total_reward is available from the last episode
        latest_loss = episode_loss  # Assuming episode_loss is available from the last episode

        agent.save()  # Calls agent.save() which saves to ACTOR_PATH and CRITIC_PATH
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'episode': last_completed_episode,
            'latest_reward': latest_reward,
            'latest_loss': latest_loss
        }, checkpoint_file)
        print(f"Progress saved to {checkpoint_file}")

        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()



