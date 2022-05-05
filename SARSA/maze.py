"""
Module to generate a maze.
"""
from typing import Tuple
import numpy as np

from .ghost import Ghost

class Maze:
    """
    Maze class.
    """

    # list of possible actions, corresponding to left, right, up, down
    actions = [0, 1, 2, 3]

    def __init__(self, shape: Tuple[int, int], nb_exits: int, nb_ghosts: int, seed: int = None):
        """
        Constructor.

        Parameters
        ----------
        shape : Tuple[int,int]
            Shape of the maze.
        nb_exits : int
            Number of exits.
        nb_ghost : int
            Number of ghosts.
        seed : int
            Seed for the random number generator.
        """
        self.shape = shape
        self.nb_exits = nb_exits
        self.nb_ghosts = nb_ghosts
        self.ghosts = []
        self.rng = np.random.default_rng(seed=seed)
        self.maze = None
        self.generate_maze()
        self.current_position = self.generate_start_position()

    def generate_maze(self) -> np.ndarray:
        """
        Generate a maze.
        """
        self.maze = np.zeros(self.shape, dtype=np.int)
        for _ in range(self.nb_exits):
            x, y = self.generate_exit()
            self.maze[x, y] = 1
        for _ in range(self.nb_ghosts):
            self.ghosts.append(self.generate_ghost())

    def generate_exit(self) -> Tuple[int, int]:
        """
        Generate an exit.

        Returns
        -------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        x, y = self.generate_random_coord()
        while self.maze[x, y] == 1:
            x, y = self.generate_random_coord()
        return x, y
    
    def generate_ghost(self) -> Tuple[int, int]:
        """
        Generate initial position of a ghost.

        Returns
        -------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        x, y = self.generate_random_coord()
        while self.maze[x, y] == 1:
            x, y = self.generate_random_coord()
        return Ghost(x, y, 0)


    def generate_start_position(self) -> Tuple[int, int]:
        """
        Generate a start position.

        Returns
        -------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        x, y = self.generate_random_coord()
        while self.maze[x, y] == 1:
            x, y = self.generate_random_coord()
        return x, y

    def generate_random_coord(self) -> Tuple[int, int]:
        """
        Generate random coordinates.

        Returns
        -------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        x = self.rng.integers(self.shape[0])
        y = self.rng.integers(self.shape[1])
        return x, y

    def valid_coordinates(self, x: int, y: int) -> bool:
        """
        Check if the coordinates are valid.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.

        Returns
        -------
        valid : bool
            Whether the coordinates are valid.
        """
        return 0 <= x < self.shape[0] and 0 <= y < self.shape[1]

    def eligible_actions(self, x: int, y: int) -> list:
        """
        Return eligible moves.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.

        Returns
        -------
        eligible_actions : list
            List of eligible moves.
        """
        moves = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
        return [i for i, (x, y) in enumerate(moves) if self.valid_coordinates(x, y)]

    def move(self, x: int, y: int, action: int) -> Tuple[int, int]:
        """
        Move in the maze.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        action : int
            Action to take.

        Returns
        -------
        x : int
            New X coordinate.
        y : int
            New Y coordinate.
        """
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
        return x, y

    def step(self, action: int) -> bool:
        """
        Step in the maze.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        done : bool
            Whether the game is done.
        """
        x, y = self.current_position
        x, y = self.move(x, y, action)
        self.current_position = x, y

        self.update_ghosts()
        
        done = self.maze[x, y] == 1
        return done

    def update_ghosts(self):
        # actions des ghosts
        for g in self.ghosts:
            g.x, g.y = self.move(g.x, g.y, g.act(maze=self))

    def get_exits(self) -> list:
        """
        Return exits.

        Returns
        -------
        exits : list
            List of exits.
        """
        return np.where(self.maze == 1)


