import matplotlib.pyplot as plt

import torch

from src.data.sample_case import *
from src.visualization.visualize import plot_test_obstacles


def plot_path(path):
    # Plot the obstacles
    plot_test_obstacles(obs_1, obs_2, obs_3, obs_4)
    plt.plot(path[:, :, 0], path[:, :, 1], '-o')
    plt.pause(0.0001)
    plt.cla()


def evaluate(model, train_loader, batch_size):
    model.eval()
    h = model.init_hidden(batch_size)
    output = []
    for _, output_data, appended_data, _ in train_loader:
        h = h.data
        model.zero_grad()
        out, h = model(appended_data, h)
        output.append(out[:, -2, :])
    output = torch.stack(output, dim=0).squeeze(1).detach().numpy()
    # Plot the obstacles
    plot_test_obstacles(obs_1, obs_2, obs_3, obs_4)
    plt.plot(output[:, 0], output[:, 1], '-o')
    plt.pause(0.0001)
    plt.cla()
