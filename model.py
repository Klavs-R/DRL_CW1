import random
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)


# Deep Q-Network class
class DQN(nn.Module):

    def __init__(self, state_n, action_n, layer_nodes):
        """
        Initialise Q-Network layers

        :param state_n (Int): Dimensions of state space
        :param action_n (Int): Dimensions of the action space
        :param layer_nodes (List[Int]): List of number of nodes in each hidden layer
        """
        super(DQN, self).__init__()

        # Initiate in and output layers
        layers = [nn.Linear(state_n, layer_nodes[0])]

        for i in range(len(layer_nodes)-1):
            layers.append(nn.Linear(layer_nodes[i], layer_nodes[i+1]))

        layers.append(nn.Linear(layer_nodes[-1], action_n))

        self.layers = nn.ModuleList(layers)

    def forward(self, state):
        """
        Mapping input state to action values through the NN

        :param state: Current state
        :return: Action values
        """

        # Pass output of each layer as input to the next
        x = f.relu(self.layers[0](state))

        for layer in self.layers[1:-1]:
            x = f.relu(layer(x))

        return self.layers[-1](x)


# Memory Buffer for Prioritized Experience Replay
class MemoryBuffer:

    def __init__(self, buffer_size):
        """
        Initialise the memory buffer

        :param buffer_size (Int): Max length of the memory buffer
        """
        self.experiences = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """
        Adding an experience to memory
        """
        # Experience tuple is initially assigned the highest priority
        experience = self.experience(state, action, reward, next_state, done)
        self.experiences.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences

        :return: List of experience tuples and accompanying weights
        """

        experiences = random.sample(self.experiences, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.experiences)


class Agent:

    def __init__(
            self,
            state_n,
            action_n,
            layer_nodes,
            batch_n=64,
            gamma=0.99,
            tau=0.001,
            learning_rate=5e-4,
            update_every=4,
    ):
        """
        Initialise agent to interact with and learn from the env

        :param state_n (Int): Dimensions of state space
        :param action_n (Int): Dimensions of the action space
        :param layer_nodes (List[Int]): List of number of nodes in each hidden layer
        """

        self.state_n = state_n
        self.action_n = action_n
        self.batch_n = batch_n
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        # Prioritised replay memory
        self.memory = MemoryBuffer(BUFFER_SIZE)

        # Q-Networks
        self.local_nn = DQN(state_n, action_n, layer_nodes).to(device)
        self.target_nn = DQN(state_n, action_n, layer_nodes).to(device)
        self.optimiser = optim.Adam(self.local_nn.parameters(), lr=learning_rate)

        # Initial time step
        self.t_step = 0

    def act(self, state, epsilon=0.0):
        """
        Return actions for given states with current policy

        :param state: Current state
        :param epsilon: Epsilon in epsilon-greedy selection
        :return: Action according to current policy
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.local_nn.eval()
        with torch.no_grad():
            action_values = self.local_nn(state)
        self.local_nn.train()

        if random.random() > epsilon:
            action_values = action_values.numpy()[0]
            options = np.where(action_values == action_values.max())[0]
            return np.random.choice(options)
        else:
            return np.random.choice(np.arange(self.action_n))

    def step(self, state, action, reward, next_state, done):
        """
        Complete updates at each time step
        """
        # First, save the last experience in memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn if necessary
        self.t_step += 1
        if self.t_step >= self.update_every:
            self.t_step -= self.t_step
            self.replay()

    def replay(self):
        """
        Replay and learn from a batch in memory
        """
        if len(self.memory) <= self.batch_n:
            return

        # Sample a batch of experiences from the memory buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_n)

        # DOUBLE Q-LEARNING ------------------------------------------------------------ #
        # Best action values according to local nn
        next_q_local = self.local_nn(next_states).detach()
        target_q_actions = torch.argmax(next_q_local, dim=1)

        # Calculate the Q-values of the next states and actions using target nn
        next_q_target = self.target_nn(next_states).detach()
        q_targets = next_q_target[range(self.batch_n), target_q_actions].unsqueeze(1)

        # Calculate the Q-targets for the current states
        q_targets = rewards + self.gamma * q_targets * (1 - dones)
        # ------------------------------------------------------------------------------ #

        # Get expected Q-values from local nn
        q_expected = self.local_nn(states).gather(1, actions)

        # Calculate and minimise loss at each experience
        loss = f.mse_loss(q_expected, q_targets)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Soft update the target network
        self.soft_update(self.local_nn, self.target_nn, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Update target model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
