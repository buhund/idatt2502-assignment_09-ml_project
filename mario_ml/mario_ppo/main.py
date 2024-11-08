# main.py

from src.environment import create_env
from src.agent import PPOAgent
from config import ENV_NAME, RENDER_MODE
from config import ACTOR_PATH, CRITIC_PATH
from src.train import load_trained_agent, run_instance
import torch


def main():
    env = create_env(map=ENV_NAME, output_path=f"output/{ENV_NAME}.mp4")

    in_dim = env.observation_space.shape
    num_actions = env.action_space.n
    agent = load_trained_agent(in_dim, num_actions, ACTOR_PATH, CRITIC_PATH)

    run_instance(agent, env)


if __name__ == "__main__":
    main()



