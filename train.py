import os
import glob

import torch

from src.data.sample_case import *

from src.datasets.torch_dataset import TorchDataset
from src.models.nets import GRUNet
from src.models.evaluate import evaluate, plot_path

from src.visualization.visualize import plot_obstacles, plot_test_obstacles
from utils import skip_run, seed_everything

# Seed everything at the beginning
seed_everything()

with skip_run('skip', 'plot_obstacles') as check, check():
    obstacles = [
        [(0, 0), (0.9, 0), (0.9, -0.5), (0.0, -0.5)],
        [(-1, 0.2), (-0.7, 0.2), (-0.7, 0.5), (-1, 0.5)],
    ]
    plot_obstacles(obstacles)
    plot_test_obstacles()

with skip_run('skip', 'testing_horizon') as check, check():
    print(phi)
    print(phi.get_horizon())
    print('-' * 32)

    for i in range(10, 30):
        x = X[0:i, :1].unsqueeze(0)
        y = X[0:i, 1:].unsqueeze(0)

        box_inputs = ((x, x), (y, y))
        circle_inputs = torch.norm(
            X[0:i, :] - obs_3[:2].unsqueeze(0), dim=-1, keepdim=True
        ).unsqueeze(0)

        horizon = phi.get_horizon()

        robustness_trace = torch.relu(
            -phi.robustness_trace((circle_inputs, box_inputs), scale=-1).squeeze()
        )

        robustness = torch.relu(
            -phi.robustness((circle_inputs, box_inputs), scale=-1).squeeze()
        )
        print(robustness)

with skip_run('skip', 'testing_horizon') as check, check():
    batch_size = 1
    learn_rate = 0.001
    input_dim = 2
    hidden_dim = 2
    output_dim = 2
    n_layers = 1

    EPOCHS = 10000

    # Dataset
    dataset = TorchDataset(points)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    model.load_state_dict(torch.load('data/linear_path_model.pth'))

    # Critiria
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of the model")
    epoch_times = []

    counter = 0
    epoch = 0
    x = None
    y = None
    cir = None
    test_out = None
    time_horizon = phi.get_horizon()
    h = model.init_hidden(batch_size)
    model.train()

    for input_data, _, appended_data, output_data in train_loader:
        avg_loss = 0.0
        epoch += 1
        counter = 0

        cir, x, y = None, None, None
        model.zero_grad()
        while True:
            counter += 1
            # model.zero_grad()
            out, h = model(appended_data, h.data)
            x = out[:, :, 0].unsqueeze(-1)
            y = out[:, :, 1].unsqueeze(-1)
            cir = out

            box_inputs = ((x, x), (y, y))
            circle_inputs = torch.norm(
                cir.squeeze(0) - obs_3[:2].unsqueeze(0), dim=-1, keepdim=True
            ).unsqueeze(0)

            # Loss with robustness
            robustness_regions = torch.relu(
                -(phi.robustness((box_inputs, circle_inputs), scale=-1).squeeze())
            )
            loss = (
                criterion(out, output_data)
                + robustness_regions
                + 2 * criterion(out[:, 0, :], -1 * torch.ones(out[:, 0, :].shape))
            )
            # print(10 * criterion(out[:, 0, :], -1 * torch.ones(out[:, 0, :].shape)))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if (robustness_regions < 1e-6) & ((avg_loss / counter) < 1e-1):
                print("FINISHED!")
                break

            plot_path(out.detach().numpy())
        print(f'Epoch {epoch}.......Average Loss for Epoch: {avg_loss}')

