import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

import numpy as np

import torch


def create_path_object(coordinates):
    path_data = []

    for i in range(len(coordinates)):
        if i == 0:
            path_data.append((Path.MOVETO, (coordinates[i])))
        else:
            path_data.append((Path.LINETO, (coordinates[i])))

    # Add the final closing path point
    path_data.append((Path.CLOSEPOLY, (coordinates[0])))

    codes, verts = zip(*path_data)
    path = Path(verts, codes)
    return path


def plot_obstacles(obstacles):
    fig, ax = plt.subplots()

    # plot each obstacle
    for obstacle in obstacles:
        path = create_path_object(obstacle)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)

    ax.grid()
    plt.axis("equal")

    plt.show()


def plot_test_obstacles(obs_1, obs_2, obs_3, obs_4):
    plt.plot(
        [obs_1[0], obs_1[0], obs_1[1], obs_1[1], obs_1[0]],
        [obs_1[2], obs_1[3], obs_1[3], obs_1[2], obs_1[2]],
        linewidth=5,
        color='#e7585b',
    )
    plt.plot(
        [obs_2[0], obs_2[0], obs_2[1], obs_2[1], obs_2[0]],
        [obs_2[2], obs_2[3], obs_2[3], obs_2[2], obs_2[2]],
        linewidth=5,
        color='#e7585b',
    )
    plt.plot(
        [obs_4[0], obs_4[0], obs_4[1], obs_4[1], obs_4[0]],
        [obs_4[2], obs_4[3], obs_4[3], obs_4[2], obs_4[2]],
        linewidth=5,
        color='#000000',
    )
    plt.plot(
        [obs_3[0] + obs_3[2].numpy() * np.cos(t) for t in np.arange(0, 3 * np.pi, 0.1)],
        [obs_3[1] + obs_3[2].numpy() * np.sin(t) for t in np.arange(0, 3 * np.pi, 0.1)],
        linewidth=5,
        color='#76b7b2',
    )

    plt.scatter([-1], [-1], s=300, c="orange")
    plt.scatter([1], [1], s=800, marker='*', c="orange")
    plt.axis("equal")
    plt.grid()
