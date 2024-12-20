import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agent import Agent
import torch
import os
from torch.utils.tensorboard import SummaryWriter  # Check model progress by running "tensorboard --logdir=runs" in terminal

# Environment and Hyperparameters
ENV_NAME = 'SuperMarioBros-1-1-v0'
SAVE_PATH = "mario_dqn_checkpoint.pth"
DISPLAY = True
NUM_OF_EPISODES = 50_000

# Initialize TensorBoard
writer = SummaryWriter(log_dir="runs/mario_dqn")

print("****************** STARTING MARIO DEEP Q NETWORK ******************")
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Load checkpoint if available
start_episode = 1
if os.path.exists(SAVE_PATH):
    checkpoint = torch.load(SAVE_PATH, weights_only=True)
    agent.online_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    agent.learn_step_counter = checkpoint['learn_step_counter']
    start_episode = checkpoint.get('episode', 1)
    print(f"Loaded previous checkpoint. Resuming from episode {start_episode}.")
else:
    print("No checkpoint found, starting fresh.")

try:
    for i in range(start_episode, NUM_OF_EPISODES + 1):
        done = False
        state, _ = env.reset()
        total_reward = 0  # Track total reward for the episode

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            
            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            total_reward += reward  # Accumulate reward for this episode

        # Log metrics to TensorBoard
        writer.add_scalar("Reward/episode", total_reward, i)
        writer.add_scalar("Epsilon/episode", agent.epsilon, i)
        print(f"Episode {i} completed with total reward: {total_reward}.")

        # Save progress every 100 episodes
        if i % 100 == 0:
            torch.save({
                'model_state_dict': agent.online_network.state_dict(),
                'target_state_dict': agent.target_network.state_dict(),
                'epsilon': agent.epsilon,
                'learn_step_counter': agent.learn_step_counter,
                'episode': i
            }, SAVE_PATH)
            print(f"Checkpoint saved at episode {i}.")

except KeyboardInterrupt:
    print("Manual interrupt detected. Saving progress before exit...")

finally:
    # Final save upon completion or interruption
    torch.save({
        'model_state_dict': agent.online_network.state_dict(),
        'target_state_dict': agent.target_network.state_dict(),
        'epsilon': agent.epsilon,
        'learn_step_counter': agent.learn_step_counter,
        'episode': i
    }, SAVE_PATH)
    print(f"Progress saved to {SAVE_PATH}")

    # Close TensorBoard writer
    writer.close()
    print("TensorBoard data saved.")

env.close()
print("Training finished, environment closed.")
