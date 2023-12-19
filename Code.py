import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import gym
from gym.wrappers import RecordVideo
from tqdm import tqdm
from collections import deque, namedtuple 
import matplotlib.pyplot as plt

# Configure device
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Neural Network for Deep Q-Learning
class QLearningNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QLearningNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)

# Define the Deep Q-Learning Agent
class ReinforcementAgent():
    def __init__(self, state_space, action_space, batch_size=64, alpha=1e-4, discount_rate=0.99, buffer_size=int(1e5), update_freq=5, tau=1e-3):
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.update_freq = update_freq
        self.tau = tau

        self.policy_net = QLearningNetwork(state_space, action_space).to(compute_device)
        self.target_net = QLearningNetwork(state_space, action_space).to(compute_device)
        self.optim = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

        self.experience_replay = MemoryBuffer(action_space, buffer_size, batch_size)
        self.update_counter = 0

    def decide_action(self, observation, epsilon):
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(compute_device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(observation)
        self.policy_net.train()

        if random.random() < epsilon:  # Use epsilon here instead of exploration_rate
            return random.choice(np.arange(self.action_space))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def store_experience(self, state, action, reward, next_state, is_done):
        self.experience_replay.add(state, action, reward, next_state, is_done)
        self.update_counter += 1
        if self.update_counter % self.update_freq == 0:
            if len(self.experience_replay) >= self.batch_size:
                experiences = self.experience_replay.sample()
                self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.discount_rate * Q_targets_next * (1 - dones))
        Q_expected = self.policy_net(states).gather(1, actions)

        loss_val = self.loss(Q_expected, Q_targets)
        self.optim.zero_grad()
        loss_val.backward()
        self.optim.step()
        self.soft_update(self.policy_net, self.target_net)

    def soft_update(self, local_model, target_model):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# Define the Replay Buffer
class MemoryBuffer():
    def __init__(self, action_dimensions, memory_capacity, batch_size):
        self.action_dimensions = action_dimensions
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "is_done"])

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, is_done):
        e = self.experience(state, action, reward, next_state, is_done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(compute_device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(compute_device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(compute_device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(compute_device)
        dones = torch.from_numpy(np.vstack([e.is_done for e in experiences if e is not None]).astype(np.uint8)).float().to(compute_device)
        return (states, actions, rewards, next_states, dones)



# Function for training the agent
def train_agent(environment, deep_q_agent, total_episodes=2000, max_steps=1000, start_epsilon=1.0, end_epsilon=0.1, epsilon_decay=0.995, desired_score=200, checkpoint=False, record_interval=100):
    score_history = []
    epsilon = start_epsilon

    # Progress bar for training
    progress_bar = tqdm(range(total_episodes), desc="Training Progress", unit="episode")
    for episode in progress_bar:
        # Setup environment for recording at specified intervals
        if episode % record_interval == 0:
            recording_env = RecordVideo(environment, video_folder=f'video/episode_{episode}')
        else:
            recording_env = environment

        state = recording_env.reset()  # Reset the environment
        episode_score = 0

        # Iterate over steps in the episode
        for step in range(max_steps):
            action = deep_q_agent.decide_action(state, epsilon)  # Decide action based on epsilon-greedy policy
            next_state, reward, done, _ = recording_env.step(action)  # Take action and observe outcome
            deep_q_agent.store_experience(state, action, reward, next_state, done)  # Store experience in replay buffer
            state = next_state  # Update state
            episode_score += reward  # Accumulate score

            if done:  # Check if episode is finished
                break

        # Append episode score to history and update progress bar
        score_history.append(episode_score)
        avg_score = np.mean(score_history[-100:])  # Calculate average score of last 100 episodes
        epsilon = max(end_epsilon, epsilon * epsilon_decay)  # Update epsilon value
        progress_bar.set_postfix_str(f"Score: {episode_score:.2f}, Avg Score: {avg_score:.2f}")

        # Check if desired score is achieved
        if len(score_history) >= 100 and avg_score >= desired_score:
            print("\nTarget score achieved!")
            break

        # Save checkpoint if required
        if checkpoint and episode % record_interval == 0:
            torch.save(deep_q_agent.policy_net.state_dict(), f'checkpoint_episode_{episode}.pth')

    return score_history


# Function for testing the agent
def test_agent(environment, deep_q_agent, num_tests=3):
    for _ in range(num_tests):
        state = environment.reset()
        total_reward = 0
        while True:
            action = deep_q_agent.decide_action(state, epsilon=0)  # Always choose the best action
            state, reward, done, _ = environment.step(action)
            total_reward += reward
            if done:
                break
        print(f"Test run score: {total_reward}")

# Function to plot the scores
def display_scores(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title("Agent's Scores Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid()
    plt.show()

# Main function to execute training and testing
def main():
    env = gym.make('LunarLander-v2')


    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    agent = ReinforcementAgent(
        state_space=state_space_dim,
        action_space=action_space_dim,
        batch_size=128,
        alpha=1e-3,
        discount_rate=0.99,
        buffer_size=10000,  # Use buffer_size instead of memory_capacity
        update_freq=5,
        tau=1e-3
    )

    training_scores = train_agent(env, agent, total_episodes=5000, desired_score=250, checkpoint=True)
    display_scores(training_scores)

    # Test the trained agent
    test_agent(env, agent, num_tests=10)

    if str(compute_device) == "cuda":
        torch.cuda.empty_cache()

# Execute the main function
if __name__ == "__main__":
    main()
