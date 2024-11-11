import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agent import Agent
import torch
import os
from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir=runs

# Environment and Hyperparameters
ENV_NAME = 'SuperMarioBros-1-1-v0'
SAVE_PATH = "mario_ddqn_checkpoint.pth"
DISPLAY = True
NUM_OF_EPISODES = 20_000

# Initialize TensorBoard
writer = SummaryWriter(log_dir="runs/mario_ddqn")

print("****************** STARTING MARIO DOUBLE DEEP Q NETWORK ******************")
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Load checkpoint if available
def load_checkpoint(agent):
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, weights_only=True)
        agent.online_network.load_state_dict(checkpoint['model_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        agent.learn_step_counter = checkpoint['learn_step_counter']
        start_episode = checkpoint.get('episode', 1)
        print(f"Loaded checkpoint. Resuming from episode {start_episode}.")
        return start_episode
    else:
        print("No checkpoint found, starting fresh.")
        return 1

# Start Training
def train():
    start_episode = load_checkpoint(agent)

    def log_metrics(episode, total_reward, max_x_pos, flag_get, time_used):
        # Log metrics to TensorBoard
        writer.add_scalar("Metrics/Total_Reward", total_reward, episode)
        writer.add_scalar("Metrics/Max_X_Position", max_x_pos, episode)
        writer.add_scalar("Metrics/Flag_Get", flag_get, episode)
        writer.add_scalar("Metrics/Time_Used", time_used, episode)

    try:
        for i in range(start_episode, NUM_OF_EPISODES + 1):
            done = False
            state, _ = env.reset()
            total_reward = 0
            max_x_pos = 0
            flag_get = 0

            while not done:
                action = agent.choose_action(state)
                new_state, reward, done, truncated, info = env.step(action)

                agent.store_in_memory(state, action, reward, new_state, done)
                agent.learn()
                state = new_state
                total_reward += reward
                max_x_pos = max(max_x_pos, info.get("x_pos", 0))

                if info.get("flag_get", False):
                    flag_get = 1

            time_used = 400 - info.get("time", 400)
            log_metrics(i, total_reward, max_x_pos, flag_get, time_used)

            print(f"Episode {i} | Total Reward: {total_reward} | Max X Position: {max_x_pos} | Flag: {flag_get} | Time Used: {time_used}")

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
        torch.save({
            'model_state_dict': agent.online_network.state_dict(),
            'target_state_dict': agent.target_network.state_dict(),
            'epsilon': agent.epsilon,
            'learn_step_counter': agent.learn_step_counter,
            'episode': i
        }, SAVE_PATH)
        print(f"Progress saved to {SAVE_PATH}")

        writer.close()
        print("TensorBoard data saved.")

    env.close()
    print("Training finished, environment closed.")

# Evaluation mode function (runs independently)
def evaluate(num_episodes=1000):
    print("Starting Evaluation Mode")
    start_episode = load_checkpoint(agent)
    agent.epsilon = 0.0  # Disable exploration for evaluation

    for i in range(1, num_episodes + 1):
        done = False
        state, _ = env.reset()
        total_reward = 0
        max_x_pos = 0
        flag_get = 0

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            state = new_state
            total_reward += reward
            max_x_pos = max(max_x_pos, info.get("x_pos", 0))

            if info.get("flag_get", False):
                flag_get = 1

        time_used = 400 - info.get("time", 400)
        print(f"Eval Episode {i} | Total Reward: {total_reward} | Max X Position: {max_x_pos} | Flag: {flag_get} | Time Used: {time_used}")

# Uncomment one of the following to either train or evaluate independently.
train()
# evaluate(num_episodes=1000)
