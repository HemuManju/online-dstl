import torch

from src.stlcg import stlcg


def inside_box(xy, obs):
    x = stlcg.Expression('x', xy[:, :1].unsqueeze(0))
    y = stlcg.Expression('y', xy[:, 1:].unsqueeze(0))
    r1 = stlcg.Expression('x1', obs[:1].unsqueeze(-1).unsqueeze(-1))
    r2 = stlcg.Expression('x2', obs[1:2].unsqueeze(-1).unsqueeze(-1))
    r3 = stlcg.Expression('y1', obs[2:3].unsqueeze(-1).unsqueeze(-1))
    r4 = stlcg.Expression('y2', obs[3:4].unsqueeze(-1).unsqueeze(-1))
    inputs = ((x, x), (y, y))
    return ((x > r1) & (x < r2)) & ((y > r3) & (y < r4)), inputs


def always_stay_outside_circle(xy, obs):
    d1 = stlcg.Expression(
        'd1', torch.norm(xy - obs[:2].unsqueeze(0), dim=-1, keepdim=True).unsqueeze(0)
    )
    r1 = stlcg.Expression('r', obs[2:3].unsqueeze(-1).unsqueeze(-1))
    return stlcg.Always(subformula=(d1 > r1)), d1
