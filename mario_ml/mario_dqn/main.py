import concurrent.futures
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from agent import Agent

# Toggle between 'rgb_array' for no display and 'human' for real-time display
#RENDER_MODE = 'rgb_array'
RENDER_MODE = 'human'

ENV_NAME = 'SuperMarioBros-1-1-v0'
NUM_INSTANCES = 1  # Set to 1 for single instance when using `human` mode, increase if `rgb_array`

def run_instance(instance_id, display=False):
    env = gym_super_mario_bros.make(ENV_NAME, render_mode=RENDER_MODE, apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)

    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

    for episode in range(100):  # Adjust number of episodes
        done = False
        state, _ = env.reset()
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)

            # Capture the frame if in 'rgb_array' mode, but do nothing with it
            if RENDER_MODE == 'rgb_array':
                current_frame = env.render()  # Returns an RGB array, can use for debugging if needed

            # Store experience and train the agent
            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()
            state = new_state

        print(f"Instance {instance_id} - Episode {episode + 1} completed.")

    env.close()

# Ensuring multiprocessing is only run in the main context
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_instance, i, display=(RENDER_MODE == 'human' and i == 0))
            for i in range(NUM_INSTANCES)
        ]