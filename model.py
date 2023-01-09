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
        self.in_layer = nn.Linear(state_n, layer_nodes[0])
        self.out_layer = nn.Linear(layer_nodes[-1], action_n)

        # Initiate all hidden layers
        self.h_layers = []

        for i in range(len(layer_nodes)-1):
            self.h_layers.append(nn.Linear(layer_nodes[i], layer_nodes[i+1]))

    def forward(self, state):
        """
        Mapping input state to action values through the NN

        :param state: Current state
        :return: Action values
        """

        # Pass output of each layer as input to the next
        x = f.relu(self.in_layer(state))

        for layer in self.h_layers:
            x = f.relu(layer(x))

        return self.out_layer(x)


# Memory Buffer for Prioritized Experience Replay
class MemoryBuffer:

    def __init__(self, buffer_size):
        """
        Initialise the prioritised memory buffer

        :param buffer_size (Int): Max length of the memory buffer
        """
        self.experiences = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """
        Adding an experience to memory
        """
        # Experience tuple is initially assigned the highest priority
        experience = self.experience(state, action, reward, next_state, done)
        max_priority = max(self.priorities, default=1)

        self.experiences.append(experience)
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        """
        Sample a batch of experiences with probability based on priority

        :return: List of experience tuples and accompanying weights
        """

        total_priority = sum(self.priorities)
        probs = [priority / total_priority for priority in self.priorities]

        # Double normalise as above is not accurate enough
        probs = np.array(probs)
        probs = probs / float(probs.sum())

        indices = np.random.choice(len(self.experiences), batch_size, p=probs, replace=False)
        experiences = [self.experiences[idx] for idx in indices]

        states = torch.Tensor([e.state for e in experiences if e is not None]).float().to(device)
        actions = torch.Tensor([e.action for e in experiences if e is not None]).long().to(device)
        rewards = torch.Tensor([e.reward for e in experiences if e is not None]).float().to(device)
        next_states = torch.Tensor([e.next_state for e in experiences if e is not None]).float().to(device)
        dones = torch.Tensor([e.done for e in experiences if e is not None]).float().to(device)

        return indices, (states, actions, rewards, next_states, dones)

    def update_priorities(self, indices, errors):
        """
        Update priorities of given experiences with given errors
        """
        for i, error in zip(indices, errors):
            self.priorities[i] = abs(float(error)) + 1e-5

    def __len__(self):
        return len(self.experiences)

# Testing non-prioritised memory replay
# class MemoryBuffer:
#
#     def __init__(self, buffer_size):
#         """
#         Initialise the prioritised memory buffer
#
#         :param buffer_size (Int): Max length of the memory buffer
#         """
#         self.experiences = deque(maxlen=buffer_size)
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#
#     def add(self, state, action, reward, next_state, done):
#         """
#         Adding an experience to memory
#         """
#         experience = self.experience(state, action, reward, next_state, done)
#         self.experiences.append(experience)
#
#     def sample(self, batch_size):
#         """
#         Sample a batch of experiences with probability based on priority
#
#         :return: List of experience tuples and accompanying weights
#         """
#
#         indices = np.random.choice(len(self.experiences), batch_size, replace=False)
#         experiences = [self.experiences[idx] for idx in indices]
#
#         states = torch.Tensor([e.state for e in experiences if e is not None]).float().to(device)
#         actions = torch.Tensor([e.action for e in experiences if e is not None]).long().to(device)
#         rewards = torch.Tensor([e.reward for e in experiences if e is not None]).float().to(device)
#         next_states = torch.Tensor([e.next_state for e in experiences if e is not None]).float().to(device)
#         dones = torch.Tensor([e.done for e in experiences if e is not None]).float().to(device)
#
#         return indices, (states, actions, rewards, next_states, dones)
#
#     def update_priorities(self, indices, errors):
#         """
#         Update priorities of given experiences with given errors
#         """
#         pass
#
#     def __len__(self):
#         return len(self.experiences)


class Agent:

    def __init__(
            self,
            state_n,
            action_n,
            layer_nodes,
            batch_n=128,
            gamma=0.99,
            tau=0.001,
            learning_rate=5e-4,
            update_every=4
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

    def step(self, state, action, reward, state2, done):
        """
        Complete updates at each time step
        """
        # First, save the last experience in memory
        self.memory.add(state, action, reward, state2, done)

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
        indices, experiences = self.memory.sample(self.batch_n)
        states, actions, rewards, next_states, dones = experiences

        # DOUBLE Q-LEARNING ------------------------------------------------------------ #
        # Best action values according to local nn
        next_q_local = self.local_nn(next_states)
        target_q_actions = torch.argmax(next_q_local, dim=1)

        # Calculate the Q-values of the next states and actions using target nn
        next_q_target = self.local_nn(next_states)
        q_targets = next_q_target[range(self.batch_n), target_q_actions]

        # Calculate the Q-targets for the current states
        q_targets = rewards + self.gamma * q_targets * (1 - dones)
        # ------------------------------------------------------------------------------ #

        # Get expected Q-values from local nn
        q_expected = self.local_nn(states)[range(self.batch_n), actions]

        # Update the priorities in the memory buffer
        errors = q_targets - q_expected
        self.memory.update_priorities(indices, errors)

        # Calculate and minimise loss at each experience
        loss = f.mse_loss(q_expected.unsqueeze(1), q_targets.unsqueeze(1))
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












