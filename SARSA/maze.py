"""
Module to generate a maze.
"""
from typing import Tuple
import numpy as np

class Maze():
    """
    Maze class.
    """

    # list of possible actions, corresponding to left, right, up, down
    actions=[0, 1, 2, 3]

    def __init__(self, shape:Tuple[int,int], exits:int, seed:int=None):
        """
        Constructor.
        """
        self.shape = shape
        self.exits = exits
        self.maze = self.generate_maze()
        self.rng = np.random.RandomState(seed=seed)
        self.current_position = self.generate_random_coord()

    def generate_maze(self) -> np.ndarray:
        """
        Generate a maze.
        """
        maze = np.zeros(self.shape, dtype=np.int)
        for _ in range(self.exits):
            x, y = self.generate_exit()
            maze[x, y] = 1
        return maze

    def generate_exit(self) -> Tuple[int, int]:
        """
        Generate an exit.
        """
        x, y = self.generate_random_coord()
        while self.maze[x, y] == 1:
            x, y = self.generate_random_coord()
        return x, y

    def generate_random_coord(self) -> Tuple[int, int]:
        """
        Generate an exit.
        """
        x = self.rng.randint(self.shape[0])
        y = self.rng.randint(self.shape[1])
        return x, y

    def valid_coordinates(self, x:int, y:int) -> bool:
        """
        Check if the coordinates are valid.
        """
        return 0 <= x < self.shape[0] and 0 <= y < self.shape[1]

    def eligible_actions(self, x:int, y:int) -> list:
        """
        Return eligible moves.
        """
        moves = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        return [i for i, (x, y) in enumerate(moves) if self.valid_coordinates(x, y)]

    def move(self, x:int, y:int, action:int) -> Tuple[int, int]:
        """
        Move in the maze.
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

    def step(self, action:int) -> Tuple[int, int, int, bool]:
        """
        Step in the maze.
        """
        x, y = self.current_position
        x, y = self.move(x, y, action)
        reward = self.maze[x, y]
        self.current_position = x, y
        done = reward == 1
        return x, y, reward, done