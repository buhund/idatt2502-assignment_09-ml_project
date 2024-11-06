import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agent import Agent
import torch
import os

ENV_NAME = 'SuperMarioBros-1-1-v0'
SAVE_PATH = "mario_ddqn_checkpoint.pth"
SHOULD_TRAIN = True
DISPLAY = True
NUM_OF_EPISODES = 50_000

print("****************** STARTING MARIO DOUBLE DEEP Q NETWORK ******************")
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Optionally load previous progress if starting from a checkpoint
if os.path.exists(SAVE_PATH):
    checkpoint = torch.load(SAVE_PATH, weights_only=True)
    agent.online_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    agent.learn_step_counter = checkpoint['learn_step_counter']
    print("Loaded previous checkpoint.")
else:
    print("No checkpoint found, starting fresh.")

try:
    for i in range(NUM_OF_EPISODES):
        done = False
        state, _ = env.reset()
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            
            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()
            state = new_state

        print(f"Episode {i + 1} completed.")

        # Save progress after each episode
        torch.save({
            'model_state_dict': agent.online_network.state_dict(),
            'target_state_dict': agent.target_network.state_dict(),
            'epsilon': agent.epsilon,
            'learn_step_counter': agent.learn_step_counter
        }, SAVE_PATH)

except KeyboardInterrupt:
    print("Manual interrupt detected. Saving progress before exit...")

finally:
    # Ensure the final save upon interruption or completion
    torch.save({
        'model_state_dict': agent.online_network.state_dict(),
        'target_state_dict': agent.target_network.state_dict(),
        'epsilon': agent.epsilon,
        'learn_step_counter': agent.learn_step_counter
    }, SAVE_PATH)
    print(f"Progress saved to {SAVE_PATH}")

env.close()
print("Training finished, environment closed.")
