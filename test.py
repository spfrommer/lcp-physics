from lcp_physics.lcp.lcp import LCPFunction
import torch

import pdb

# I think things are batched?

# mass = 1.0
# mu = 1.0
# vk = torch.tensor([0.0, 0]).unsqueeze(1)
# mg = torch.tensor([0.0, mass * 1]).unsqueeze(1)
# u = torch.tensor([0.0, 0]).unsqueeze(1)

# M = mass * torch.eye(2)
# q = (-M) * vk + mg + u # This is the p argument

# G = torch.tensor([[0.0, 1, 0],
                  # [1, 0, 0],
                  # [-1, 0, 0],
                  # [0, 0, 0]])
# F = torch.tensor([[0.0, 0, 0, 0],
                  # [0, 0, 0, 1],
                  # [0, 0, 0, 1],
                  # [mu, -1, -1, 0]])
# A = torch.tensor([[]])
# m = torch.zeros(4, 1) # This is the h argument

# b = torch.tensor([])

mass = 1.0
mu = 1.0
vk = torch.tensor([0.0, 0]).unsqueeze(1).unsqueeze(0)
mg = torch.tensor([0.0, -mass * 1]).unsqueeze(1).unsqueeze(0)
u = torch.tensor([-3.0, -0.5]).unsqueeze(1).unsqueeze(0) 

M = (mass * torch.eye(2)).unsqueeze(0)
# This is the p argument
q = (torch.bmm(-M, vk) + mg - u).squeeze(2)

G = torch.tensor([[0.0, 1],
                  [1, 0],
                  [-1, 0],
                  [0, 0]]).unsqueeze(0) 
F = torch.tensor([[0.0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [mu, -1, -1, 0]]).unsqueeze(0) 
#A = torch.tensor([[]]).unsqueeze(0) 
A = torch.tensor([])
m = torch.zeros(4).unsqueeze(0) # This is the h argument

b = torch.tensor([])

lcp_solver = LCPFunction()
x = lcp_solver(M, q, G, m, A, b, F)
print(x)
