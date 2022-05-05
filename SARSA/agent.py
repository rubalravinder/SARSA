"""
Agent class for the SARSA algorithm.
"""

from abc import ABC, abstractmethod

import numpy as np

from maze import Maze

start_Q = 0
start_Q_for_exits = 0


class Agent(ABC):
    """Abstract agent class."""

    def __init__(self, maze: Maze, seed=None):
        """
        Initialize the agent.

        Parameters
        ----------
        maze : maze.Maze
            The environment to interact with.
        seed : int
            The seed for the random number generator.
        """
        self.maze = maze
        self.rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def act(self) -> int:
        """
        Choose an action.

        Returns
        -------
        action : int
            The action to take.
        """
        pass

    @abstractmethod
    def learn(self, reward, state, action, state_prime, action_prime):
        """
        Update the agent's knowledge of the environment.

        Parameters
        ----------
        action : int
            The action taken.
        reward : float
            The reward received.
        """
        pass


class RandomAgent(Agent):
    """Random agent."""

    def act(self):
        eligible_actions = self.maze.eligible_actions(*self.maze.current_position)
        return self.rng.choice(eligible_actions)

    def learn(self, reward, state, action, state_prime, action_prime):
        pass


class SARSAAgent(Agent):
    """SARSA agent."""

    def __init__(self, maze: Maze, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        """
        Initialize the agent.

        Parameters
        ----------
        maze : maze.Maze
            The environment to interact with.
        alpha : float
            The learning rate.
        gamma : float
            The discount factor.
        epsilon : float
            The probability of choosing a random action.
        seed : int
            The seed for the random number generator.
        """
        super().__init__(maze, seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = None
        self.initialize_Q()

    def initialize_Q(self):
        """
        Initialize the Q matrix.
        """
        self.Q = np.ones((self.maze.shape[0], self.maze.shape[1], 4), dtype=np.float) * start_Q

        # set exits to some value
        exits = self.maze.get_exits()
        self.Q[exits] = start_Q_for_exits

    def act(self) -> int:
        """
        Choose an action with e-greedy.

        Returns
        -------
        action : int
            The action to take.
        """

        # check eligible moves.
        eligible_actions = self.maze.eligible_actions(*self.maze.current_position)

        # decide whether to take a random action or not
        if self.rng.uniform() < self.epsilon:
            action = self.rng.choice(eligible_actions)
        else:
            action_values = self.Q[self.maze.current_position[0], self.maze.current_position[1], :]
            max_value_actions = np.nonzero(action_values == np.max(action_values[eligible_actions]))[0]
            max_value_actions = max_value_actions[np.in1d(max_value_actions, eligible_actions)]
            action = self.rng.choice(max_value_actions)

        return action

    def learn(self, reward, state, action, state_prime, action_prime):
        """
        Update the agent's knowledge of the environment.

        Parameters
        ----------
        action : int
            The action taken.
        reward : float
            The reward received.
        """

        # get current and previous positions
        x_new, y_new = state_prime
        x_old, y_old = state

        # update knowledge
        QSA_prime = self.Q[x_new, y_new, action_prime]
        QSA = self.Q[x_old, y_old, action]
        self.Q[x_old, y_old, action] += self.alpha * (reward + self.gamma * QSA_prime - QSA)
