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

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.lcp_solver = LCPFunction()
        self.mass = torch.nn.Parameter(torch.tensor([5.0]))

    def forward(self, prev_vels):
        for i, prev_vel in enumerate(prev_vels):
            vk = prev_vels[i]
            vk = vk.unsqueeze(1).unsqueeze(0)
            
            mass = self.mass
            g = torch.tensor([0.0, -1])
            mu = 1.0
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
                              [mu, -1, -1, 0]]).unsqueeze(0) 
            A = torch.tensor([])
            # This is the h argument
            m = torch.zeros(4).unsqueeze(0)

            b = torch.tensor([])

            return self.lcp_solver(M, q, G, m, A, b, F)

    # def forward(self, prev_vels):
        # next_vels = torch.zeros_like(prev_vels)
        # for i, prev_vel in enumerate(prev_vels):
            # prev_vel = prev_vels[i]
            # M, q, G, m, A, b, F = self.make_matrices(prev_vel)
            # next_vels[i, :] = self.lcp_solver(M, q, G, m, A, b, F)

        # return next_vels
    
    # def make_matrices(self, vk):
        # vk = vk.unsqueeze(1).unsqueeze(0)
        
        # mass = self.mass
        # g = torch.tensor([0.0, -1])
        # mu = 1.0
        # mg = (mass * g).unsqueeze(1).unsqueeze(0)
        # u = torch.tensor([2.0, 0]).unsqueeze(1).unsqueeze(0) 

        # M = (mass * torch.eye(2)).unsqueeze(0)
        # # This is the p argument
        # q = (torch.bmm(-M, vk) - mg - u).squeeze(2)

        # G = torch.tensor([[0.0, -1],
                          # [1, 0],
                          # [-1, 0],
                          # [0, 0]]).unsqueeze(0) 
        # F = torch.tensor([[0.0, 0, 0, 0],
                          # [0, 0, 0, 1],
                          # [0, 0, 0, 1],
                          # [mu, -1, -1, 0]]).unsqueeze(0) 
        # A = torch.tensor([])
        # # This is the h argument
        # m = torch.zeros(4).unsqueeze(0)

        # b = torch.tensor([])

        # return M, q, G, m, A, b, F

net = TestNet()

# prev_vels = torch.tensor([[1.0, 0],
                          # [0, 0],
                          # [-1, 0]])
# next_vels = torch.tensor([[2.0, 0],
                          # [1, 0],
                          # [0, 0]])
prev_vels = torch.tensor([[1.0, 0]])
next_vels = torch.tensor([[2.0, 0]])


loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

for epoch in range(500):
    # Zero the gradients
    optimizer.zero_grad()

    pred_vels = net(prev_vels)
    #print(pred_vels)

    loss = loss_func(pred_vels, next_vels)
    print('epoch: ', epoch,' loss: ', loss.item(), ' mass: ', net.mass.item())
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    #optimizer.step()
    for p in net.parameters():
        p.data.add_(0.5, p.grad.data)

    net.mass = torch.nn.Parameter(net.mass.clone().detach())
    net.lcp_solver = LCPFunction()
