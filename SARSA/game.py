"""Module to run a maze game."""

from pickletools import optimize
from time import sleep
from turtle import done

import numpy as np
from IPython.display import clear_output
from tqdm import tqdm

from .agent import Agent
from .maze import Maze
from .vizualiser import plot_maze
from .replay_buffer import ReplayBuffer

from tensorflow import keras
from tensorflow.keras import layers


class Game:
    """
    Game class.
    """

    def __init__(self, maze: Maze, agent: Agent, max_steps=100) -> None:
        """
        Constructor.

        Parameters
        ----------
        maze : maze.Maze
            The maze to play the game on.
        agent : agent.Agent
            The agent to play the game with.
        max_steps : int
            The maximum number of steps to take.
        """
        self.maze = maze
        self.agent = agent
        self.max_steps = max_steps
        self.rewards = []
        self.replay_buffer = ReplayBuffer()
        self.min_len_buffer = 1024
        self.gamma = 0.9
        self.model = keras.Sequential(
                                [layers.Dense(64, activation='relu'),
                                layers.Dense(38, activation='relu'),
                                layers.Dense(4, activation='softmax')])
        self.model.compile(loss='rmse', optimizer='adam', metrics=['rmse'])



    def run_game(self, plot: bool = False) -> None:
        """
        Run a game.

        Parameters
        ----------
        plot : bool
            Whether to plot the game.

        Returns
        -------
        reward : float
            The reward received.
        """
        self.maze.current_position = self.maze.generate_start_position()
        reward = 0
        steps = 0
        if plot:
            plot_maze(self.maze)
            sleep(1)

        # choisir une action a depuis s en utilisant la politique spécifiée par Q (par exemple ε-greedy)
        action = self.agent.act()

        # initialiser l'état s
        state = self.maze.current_position

        # répéter jusqu'à ce que s soit l'état terminal 
        maze_done = False
        agent_done = False
        ghost_done = False
        while not (maze_done or agent_done or ghost_done):
            # exécuter l'action a
            maze_done = self.maze.step(action)

            # check ghosts positions
            ghost_done = self.maze.current_position in [(g.x, g.y) for g in self.maze.ghosts]

            # observer la récompense r et l'état s'
            if not maze_done:
                reward -= 1
            steps += 1
            state_prime = self.maze.current_position
            agent_done = self.max_steps - steps <= 0

            # choisir une action a' depuis s' en utilisant la politique spécifiée par Q (par exemple ε-greedy)
            action_prime = self.agent.act()

            # Q[s, a] := Q[s, a] + α[r + γQ(s', a') - Q(s, a)]

            self.replay_buffer.add_expce(state, action, reward, state_prime, maze_done or agent_done or ghost_done)
            self.agent.learn(reward, state, action, state_prime, action_prime)

            # s ← s'
            # a ← a'
            state = state_prime
            action = action_prime

            if plot:
                # plot the current step
                clear_output(wait=True)
                plot_maze(self.maze)
                sleep(1)

        if len(self.replay_buffer) > self.min_len_buffer: # si on a assez de expces est suffisante
            lst_state, lst_action, lst_reward, lst_state_prime, lst_done = self.replay_buffer.get_batch()
        
        lst_q     = []
        for state, action, reward, state_prime, done in zip(lst_state, lst_action, lst_reward, lst_state_prime, lst_done):
            
            lst_q.append(self.model(state))

            if done :
                lst_q[-1][action] = reward
            else:
                expected_reward = self.model(state_prime)
                idx_max = np.argmax(expected_reward)
                lst_q[-1][action] = reward + expected_reward[idx_max] * self.gamma

        self.model.fit(x=lst_state, y=lst_q)
        
        return reward

    def train_agent(self, episodes: int) -> None:
        """
        Train an agent.
        """
        for _ in tqdm(range(episodes), desc="episodes"):
            reward = self.run_game(plot=False)
            self.rewards.append(reward)
