from lcp_physics.lcp.lcp import LCPFunction
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

# prev_vels = torch.tensor([[1.0, 0],
                          # [0, 0],
                          # [-1, 0],
                          # [-4, 0]])
# next_vels = torch.tensor([[2.0, 0],
                          # [1, 0],
                          # [0, 0],
                          # [-1, 0]])
# prev_vels = torch.tensor([[-1.0, 0],
                          # [-0.5, 0],
                          # [0.5, 0],
                          # [1.0, 0]])
# next_vels = torch.tensor([[0.0, 0],
                          # [0.0, 0],
                          # [0.0, 0],
                          # [0.0, 0]])
prev_vels = torch.tensor([[1.0, 0],
                          [2.0, 0],
                          [3.0, 0],
                          [4.0, 0],
                          [5.0, 0],
                          [-6.0, 0],
                          [-5.0, 0],
                          [-4.0, 0]])
next_vels = torch.tensor([[2.0, 0],
                          [3.0, 0],
                          [4.0, 0],
                          [5.0, 0],
                          [6.0, 0],
                          [-3.0, 0],
                          [-2.0, 0],
                          [-1.0, 0]])

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.lcp_solver = LCPFunction()
        # If mass starts over 2, fails to converge, 0<x<2 is good
        self.mass = torch.nn.Parameter(torch.tensor([1.0]))
        self.mass.requires_grad = False
        # Mu needs to be < 5 to converge (assuming mass fixed)
        self.mu = torch.nn.Parameter(torch.tensor([1.3]))

    def forward(self, prev_vels):
        next_vels = torch.zeros_like(prev_vels)
        for i, prev_vel in enumerate(prev_vels):
            prev_vel = prev_vels[i]
            M, q, G, m, A, b, F = self.make_matrices(prev_vel)
            next_vels[i, :] = self.lcp_solver(M, q, G, m, A, b, F)

        return next_vels
    
    def make_matrices(self, vk):
        vk = vk.unsqueeze(1).unsqueeze(0)
        
        mass = self.mass
        mu = self.mu

        g = torch.tensor([0.0, -1])
        mg = (mass * g).unsqueeze(1).unsqueeze(0)
        u = torch.tensor([2.0, 0]).unsqueeze(1).unsqueeze(0) 

        M = (mass * torch.eye(2)).unsqueeze(0)
        # This is the p argument
        q = (torch.bmm(-M, vk) - mg - u).squeeze(2)

        G = torch.tensor([[0.0, -1],
                          [1, 0],
                          [-1, 0],
                          [0, 0]]).unsqueeze(0) 
        F = torch.tensor([[0.0, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 0, 1],
                          [0, -1, -1, 0]]) + \
            mu * torch.tensor([[0.0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [1, 0, 0, 0]])
        F = F.unsqueeze(0)

        A = torch.tensor([])
        # This is the h argument
        m = torch.zeros(4).unsqueeze(0)

        b = torch.tensor([])

        return M, q, G, m, A, b, F

net = TestNet()

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.5)

for epoch in range(500):
    # Zero the gradients
    optimizer.zero_grad()

    pred_vels = net(prev_vels)
    #print(pred_vels)

    loss = loss_func(pred_vels, next_vels)
    print('epoch: ', epoch,' loss: ', loss.item(), ' mass: ', net.mass.item(), ' mu: ', net.mu.item())
    # print('epoch: ', epoch,' loss: ', loss.item(), ' mu: ', net.mu.item())
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    # optimizer.step()
    for p in net.parameters():
        if p.requires_grad:
            p.data.add_(0.1, -p.grad.data)
    
    # Needed to recreate the backwards graph
    # TODO: fix this properly
    net.lcp_solver = LCPFunction()
