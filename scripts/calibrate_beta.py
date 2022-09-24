"""
Calibrate oracle noise parameter beta on a per-task basis.
"""
import argparse
import torch
from torch.cuda import is_available
from numpy import array
from scipy.spatial.distance import pdist
from rlutils import build_params


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
parser.add_argument("oracle", type=str)
args = parser.parse_args()

device_ = torch.device("cuda" if is_available() else "cpu")
P = build_params([f"oracle.fastjet.{args.task}.{args.oracle}"], root_dir="config")
offline_graph = torch.load(P["pbrl"]["offline_graph_path"], map_location=device_)
diffs = torch.tensor(pdist(array(offline_graph.oracle_returns).reshape(-1,1))).unsqueeze(1)

# Aim is to calibrate beta for a target mistake probability
p_target = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.45])

beta = diffs.mean() * torch.rand((1, len(p_target)))
beta.requires_grad = True
opt = torch.optim.Adam([beta], lr=0.1)

while True:
    p_mean = (1. / (1. + (diffs / beta).exp())).mean(dim=0)
    loss = torch.linalg.norm(p_mean - p_target)
    print(beta.detach().numpy(), p_mean.detach().numpy(), loss.item())
    if loss < 2e-5: break
    loss.backward()
    opt.step()
    opt.zero_grad()

print(list(beta.detach().numpy()[0]))
