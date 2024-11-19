# src/agent.py

import csv
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from src.environment import create_env
from src.network import CNNNetwork
from src.utils.unique_filename import get_unique_filename
from config import ENV_NAME, RENDER_MODE, DEVICE, WEIGHTS_PATH
from config import ACTOR_PATH, CRITIC_PATH


class PPOAgent:
    def __init__(
            self,
            input_dimensions,
            num_actions,
            learning_rate=1e-4,
            gamma=0.90,
            epsilon=0.2, # Clip rate
            lambda_param=0.98,
            entropy_coefficient=0.01,
    ):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.entropy_coefficient = entropy_coefficient
        self.device = DEVICE
        self.actor = CNNNetwork(input_dimensions, out_dim=num_actions).to(self.device)
        self.critic = CNNNetwork(input_dimensions, out_dim=1).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        print(f"Running on: {self.device}")

    def select_action(self, state):

        state = state.to(self.device)
        with torch.no_grad():
            action_logits = self.actor(state)
            value = self.critic(state).squeeze(-1)
            action_probabilities = F.softmax(action_logits, dim=-1)
            action_distribution = Categorical(action_probabilities)
            action = action_distribution.sample()
            log_probability = action_distribution.log_prob(action)
        return action.item(), log_probability, value

    def compute_gae(self, rewards, values, next_value, dones):

        values = values + [next_value]
        gae = 0
        advantages = []
        returns = []

        for step in reversed(range(len(rewards))):
            delta = (
                    rewards[step]
                    + self.gamma * values[step + 1] * (1 - dones[step])
                    - values[step]
            )
            gae = delta + self.gamma * self.lambda_param * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return torch.tensor(advantages, device=self.device), torch.tensor(
            returns, device=self.device
        )

    def update(self, states, actions, old_log_probs, returns, advantages):
        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(4):  # Multiple epochs for each batch
            logits = self.actor(states)
            dist = Categorical(F.softmax(logits, dim=-1))
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Calculate PPO clipped objective
            ratios = torch.exp(log_probs - old_log_probs)  # r_t(theta) = exp(log_pi - log_pi_old)
            surr1 = ratios * advantages  # r_t * A
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()  # PPO clipped objective
            actor_loss -= self.entropy_coefficient * entropy  # Add entropy bonus to encourage exploration

            # Update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic loss for value update
            values = self.critic(states).squeeze(-1)
            critic_loss = F.mse_loss(values, returns) # Mean Squared Error for value function

            # Update critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Accumulated losses for tracking
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        # Average losses over the epochs
        avg_actor_loss = total_actor_loss / 4 # Loss is averaged over the 4 training epochs in each batch
        avg_critic_loss = total_critic_loss / 4 # Loss is averaged over the 4 training epochs in each batch

        return avg_actor_loss, avg_critic_loss

    def save(self, path=WEIGHTS_PATH): # WEIGHTS_PATH = src/weights
        os.makedirs(path, exist_ok=True)
        actor_path = os.path.join(path, "actor.pth") # Save Actor to path src/weights/actor.pth
        critic_path = os.path.join(path, "critic.pth") # Save Critic to path src/weights/critic.pth
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Model weights saved.\n"
              f"Actor: {actor_path}\n"
              f"Critic: {critic_path}")

    def train(
            self, env, num_episodes, path="training_result", output=None, render=False
    ):

        os.makedirs(path, exist_ok=True)

        if output:
            output_file = get_unique_filename(path, output)
            with open(os.path.join(path, output_file), mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward", "Got_Flag"])

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
            episode_reward = 0
            got_the_flag = False

            while not done:
                state_tensor = (
                    torch.tensor(state, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                action, log_prob, value = self.select_action(state_tensor)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                states.append(state_tensor)
                actions.append(torch.tensor(action, device=self.device))
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                dones.append(done)

                state = next_state
                if info.get("flag_get", False):
                    got_the_flag = True

                if render:
                    env.render()

            next_state_tensor = (
                torch.tensor(next_state, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                next_value = self.critic(next_state_tensor).squeeze(-1)

            advantages, returns = self.compute_gae(rewards, values, next_value, dones)
            states = torch.cat(states)
            actions = torch.stack(actions)
            old_log_probs = torch.stack(log_probs)
            returns = returns.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            self.update(states, actions, old_log_probs, returns, advantages)

            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

            if output:
                with open(
                        os.path.join(path, output_file), mode="a", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [episode + 1, episode_reward, 1 if got_the_flag else 0]
                    )


    def save_checkpoint(self, path, episode):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'episode': episode
        }
        torch.save(checkpoint, os.path.join(path, f"checkpoint_{episode}.pth"))
        print(f"Checkpoint saved at episode {episode}")


    def load_checkpoint(self, path, episode):
        checkpoint_path = os.path.join(path, f"checkpoint_{episode}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Checkpoint loaded from episode {episode}")


    def update_loss_tracking(self, states, actions, old_log_probs, returns, advantages):
        """Wrapper around update to track and return actor and critic losses."""
        avg_actor_loss, avg_critic_loss = self.update(states, actions, old_log_probs, returns, advantages)
        return avg_actor_loss, avg_critic_loss



def main():

    env = create_env(map=ENV_NAME)
    sample_observation = env.reset()
    print("Sample observation shape:", sample_observation.shape)
    agent = PPOAgent(env.observation_space.shape, env.action_space.n)

    agent.actor.load_state_dict(
        torch.load(ACTOR_PATH, map_location=torch.device("cpu"))
    )
    agent.critic.load_state_dict(
        torch.load(CRITIC_PATH, map_location=torch.device("cpu"))
    )

    agent.train(env, 10, render=RENDER_MODE, output=f"{ENV_NAME}.csv")
    agent.save()


if __name__ == "__main__":
    main()



