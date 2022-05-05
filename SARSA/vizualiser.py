import matplotlib.pyplot as plt
import numpy as np

from maze import Maze


def plot():
    """Dev function to plot the maze."""
    plt.figure(1, figsize=(12, 10), dpi=100)
    plt.axis([0, 1, 0, 1])
    plt.title("The Amazing Maze")
    plt.yticks(np.arange(0, y_shape + 1, 1))
    plt.xticks(np.arange(0, x_shape + 1, 1))
    plt.grid(color="r", linestyle="-", linewidth=1)
    plt.scatter(
        x=[0.5, 3.5], y=[0.5, 3.5], s=130, c="yellow", marker="*", edgecolors="green"
    )


def plot_maze(maze: Maze):
    """Plot the maze."""
    x_shape, y_shape = maze.shape
    exits_x, exits_y = maze.get_exits()
    agent_x, agent_y = maze.current_position

    plt.figure(1, figsize=(12, 10))
    plt.axis([0, 1, 0, 1])
    plt.yticks(np.arange(0, y_shape + 1, 1))
    plt.xticks(np.arange(0, x_shape + 1, 1))
    plt.grid(color="r", linestyle="-", linewidth=1)

    # add exits
    plt.scatter(
        x=exits_x + 0.5,
        y=exits_y + 0.5,
        s=250,
        c="yellow",
        marker="*",
        edgecolors="green",
    )

    # add agent
    plt.scatter(
        x=agent_x + 0.5,
        y=agent_y + 0.5,
        s=130,
        c="black",
        marker="o",
        edgecolors="blue",
    )

    # add ghosts
    for g in maze.ghosts:
        plt.scatter(
            x= g.x + 0.5,
            y= g.y + 0.5,
            s=130,
            c="red",
            marker="+",
            edgecolors="yellow",
        )

    plt.show()


def plot_cumsum(game):
    """Plot the cumulative sum of rewards."""
    plt.figure(1, figsize=(12, 10))
    plt.plot(np.cumsum(game.rewards), "+-")
    plt.title("Cumulative sum of rewards")
    plt.grid()
    plt.show()


def plot_cumav(game):
    """Plot the cumulative average of rewards."""
    plt.figure(1, figsize=(12, 10))
    plt.plot(np.cumsum(game.rewards) / np.arange(1, len(game.rewards) + 1), "+-")
    plt.title("Cumulative average of rewards")
    plt.grid()
    plt.show()


def plot_rollav(game, w: int = 5):
    """Plot the rolling average of rewards."""
    plt.figure(1, figsize=(12, 10))
    plt.plot(np.convolve(game.rewards, np.ones(w), "valid") / w)
    plt.title("Rolling average of rewards")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    matrice_jeu = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    matrice_jeu = np.array(matrice_jeu)
    x_shape, y_shape = matrice_jeu.shape

    plot_maze()
    plt.show()
