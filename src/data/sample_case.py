import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np


from src.stlcg import Expression, STL_Formula, Always, And

import pickle
import torch
import pickle


def intermediates(p1, p2, nb_points=8):
    """ "Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return np.asarray(
        [
            [p1[0] + i * x_spacing, p1[1] + i * y_spacing]
            for i in range(1, nb_points + 1)
        ]
    )


def inside_box(xy, obs):
    x = Expression('x', xy[:, :1].unsqueeze(0))
    y = Expression('y', xy[:, 1:].unsqueeze(0))
    r1 = Expression('x1', obs[:1].unsqueeze(-1).unsqueeze(-1))
    r2 = Expression('x2', obs[1:2].unsqueeze(-1).unsqueeze(-1))
    r3 = Expression('y1', obs[2:3].unsqueeze(-1).unsqueeze(-1))
    r4 = Expression('y2', obs[3:4].unsqueeze(-1).unsqueeze(-1))
    inputs = ((x, x), (y, y))
    return ((x > r1) & (x < r2)) & ((y > r3) & (y < r4)), inputs


def always_outside_box(xy, obs):
    is_it_inside_box, _ = inside_box(xy, obs)
    return Always(subformula=~is_it_inside_box)


def always_inside_box(xy, obs):
    is_it_inside_box, _ = inside_box(xy, obs)
    return Always(subformula=is_it_inside_box, interval=[15, 20])


def always_stay_outside_circle(xy, obs):
    d1 = Expression(
        'd1', torch.norm(xy - obs[:2].unsqueeze(0), dim=-1, keepdim=True).unsqueeze(0)
    )
    r1 = Expression('r', obs[2:3].unsqueeze(-1).unsqueeze(-1))
    return (
        Always(subformula=(d1 > r1 + 0.1), interval=[20, 30], robustness_interval=1),
        d1,
    )


# Data and obstacle creation
obs_1 = torch.tensor([0.65, 0.9, 0.60, 0.0]).float()  # red box in bottom right corner
obs_2 = torch.tensor([-0.5, 0.2, 0.8, 1.2]).float()  # green box in top right corner
obs_3 = torch.tensor([0.0, 0.1, 0.3]).float()  # blue circle in the center
obs_4 = torch.tensor([-1.0, -0.7, -0.2, 0.5]).float()  # orange box on the left

# Data
dx = 0.04
n = np.stack([np.arange(-1.16, 1.16, dx), np.arange(-1.16, 1.16, dx)]).T
x0 = -np.ones(2)
xf = np.ones(2)
N = n.shape[0]

# Tensor data
points = intermediates([-1.16, -1.16], [1.16, 1.16], nb_points=N)
X = torch.from_numpy(points).float().requires_grad_(True)

# Constraints
inside_box_1 = always_outside_box(X[:, :], obs_1)
inside_box_4 = always_inside_box(X[:, :], obs_4)
always_stay_outside_circle_formula, _ = always_stay_outside_circle(X[0:4, :], obs_3)
always_outside_box = always_outside_box(X[:, :], obs_1)
has_been_inside_box_4 = Always(
    subformula=inside_box_4, interval=[15, 20], robustness_interval=10
)

# STL formulation
phi = And(subformula1=inside_box_4, subformula2=always_stay_outside_circle_formula)
# phi = always_stay_outside_circle_formula & inside_box_4
phi = inside_box_4
phi = inside_box_4 & always_stay_outside_circle_formula
