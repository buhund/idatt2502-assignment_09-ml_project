# src/ppo/ppo_agent.py

import csv
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from src.environment import create_env
from src.network import CNNNetwork
from src.utils import get_unique_filename
from config import ENV_NAME, RENDER_MODE, DEVICE
from config import ACTOR_PATH, CRITIC_PATH


class PPOAgent:
    def __init__(
            self,
            input_dimensions,
            num_actions,
            learning_rate=3e-4,
            gamma=0.99,
            epsilon=0.2, # Standard choice for PPO epsilon.
            lambda_param=0.95,
            entropy_coefficient=0.01,
    ):
        """Initializes the PPOAgent with actor and critic network weights, along with hyperparameters for training.
            Weights are saved between runs, thus preserving the learned behaviors from one run to the next, allowing
            the agent to start from what was learned earlier, instead of from scratch every time.

        Args:
            input_dimensions (tuple): Input dimensions for the CNN, representing the state space.
            num_actions (int): Number of possible actions in the action space.
            learning_rate (float): Learning rate for both actor and critic optimizers.
            gamma (float): Discount factor for future rewards, controlling the importance of future rewards.
            epsilon (float): Clipping parameter for PPO, controlling the degree of policy update restriction.
                            In PPO, epsilon is a constant which defines a threshold for clipping during policy updates.
                            With epsilon set to 0.2, this means that each policy update is restricted to within ±20% of 
                            the old policy’s action probabilities. This prevents overly aggressive updates that might 
                            otherwise lead to suboptimal behavior.
            lambda_param (float): GAE (Generalized Advantage Estimation) lambda parameter, used in advantage calculation.
            entropy_coefficient (float): Coefficient for entropy regularization, encouraging exploration in the policy.

        Attributes:
            actor (torch.nn.Module): The neural network representing the policy (actor).
            critic (torch.nn.Module): The neural network representing the value function (critic).
            actor_optimizer (torch.optim.Optimizer): Optimizer for the actor network.
            critic_optimizer (torch.optim.Optimizer): Optimizer for the critic network.
            device (torch.device): Device to run computations on (GPU if available, else CPU).

        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.entropy_coefficient = entropy_coefficient
        self.device = DEVICE
        self.actor = CNNNetwork(input_dimensions, out_dim=num_actions).to(self.device)
        self.critic = CNNNetwork(input_dimensions, out_dim=1).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        print(f"Training on: {self.device}")

    def select_action(self, state):
        """Selects an action based on the current policy, returning both the action and its log probability.

        Args:
            state (torch.Tensor): The current state observation.

        Returns:
            action (int): The selected action.
            log_prob (torch.Tensor): Log probability of the selected action.
            value (torch.Tensor): Estimated value of the current state from the critic.

        """
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            value = self.critic(state).squeeze(-1)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def compute_gae(self, rewards, values, next_value, dones):
        """Calculates the advantages and returns using Generalized Advantage Estimation (GAE).

        Formula:
            delta_t = reward + gamma * V(s') - V(s)
            advantage_t = delta_t + gamma * lambda * advantage_t+1

        Args:
            rewards (list): Collected rewards from the episode.
            values (list): State values collected from the critic for each step in the episode.
            next_value (float): Value of the next state from the critic.
            dones (list): Boolean indicators for each step, where True indicates the end of an episode.

        Returns:
            advantages (torch.Tensor): Calculated advantages for each step in the episode.
            returns (torch.Tensor): Target returns for each step in the episode.

        """
        values = values + [next_value]
        gae = 0
        advantages = []
        returns = []

        for step in reversed(range(len(rewards))):
            # Calculate delta_t for advantage estimation
            delta = (
                    rewards[step]
                    + self.gamma * values[step + 1] * (1 - dones[step])
                    - values[step]
            )
            # GAE formula for calculating advantage estimate
            gae = delta + self.gamma * self.lambda_param * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return torch.tensor(advantages, device=self.device), torch.tensor(
            returns, device=self.device
        )

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Performs a PPO update on the actor and critic networks.

        PPO Loss Formula:
            L_clip = E[min(r_t(theta) * A, clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A)]
            Critic loss = (returns - V(s))^2
            Total loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

        Args:
            states (torch.Tensor): Batch of observed states.
            actions (torch.Tensor): Batch of actions taken.
            old_log_probs (torch.Tensor): Log probabilities of actions taken, from previous policy.
            returns (torch.Tensor): Computed returns for each state-action pair.
            advantages (torch.Tensor): Computed advantages for each state-action pair.

        """
        for _ in range(4):  # Multiple epochs for each batch
            logits = self.actor(states)
            dist = Categorical(F.softmax(logits, dim=-1))
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Calculate PPO clipped objective
            ratios = torch.exp(
                log_probs - old_log_probs
            )  # r_t(theta) = exp(log_pi - log_pi_old)
            surr1 = ratios * advantages  # r_t * A
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()  # PPO clipped objective
            actor_loss -= self.entropy_coefficient * entropy

            # Update actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Critic loss for value update
            values = self.critic(states).squeeze(-1)
            critic_loss = F.mse_loss(
                values, returns
            )  # Mean Squared Error for value function

            # Update critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def save(self, path="weights"):
        """Saves the actor and critic networks to specified files.

        Args:
            path (str): Path to save the weights weights. Will create directory if it doesn’t exist.

        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        print(f"Model weights saved to {path}")

    def train(
            self, env, num_episodes, path="training_result", output=None, render=False
    ):
        """Trains the agent in the environment using PPO.

        Args:
            env (gym.Env): The environment in which the agent will be trained.
            num_episodes (int): Number of episodes to train the agent.
            path (str): Path to save training results as a CSV file.
            output (str): Filename for the output CSV file. If file exists, a new unique name is created.
            render (bool): If True, renders the environment after each action.

        """
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


def main():
    """Sets up the environment and agent, then trains the agent.

    Loads saved weights weights if available, then trains the agent and saves the results.

    Environment:
        Super Mario Bros environment is created using the `create_env` function.

    Agent:
        Loads actor and critic networks from saved weights if available, trains for 10 episodes, then saves.
    """
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



