import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class LQRSolver(nn.Module):
    def __init__(self, dim_x, dim_u):
        super().__init__()
        # Dynamics matrices (discrete case)
        self.A = torch.eye(dim_x)
        self.B = torch.zeros(dim_x, dim_u)

        # LQR weight matrices
        self.Q = torch.eye(dim_x)
        self.R = torch.eye(dim_u)

        # LQR Parameter matrix
        self.P_sqrt = Parameter(torch.eye(dim_x))

    def forward(self, x_ic):
        K = self.compute_gain()
        u = -torch.matmul(K, x_ic)
        return u

    def compute_gain(self):
        # Discrete Ricatti equation variable
        P = torch.matmul(self.P_sqrt, self.P_sqrt.T)
        # Solve for P
        A = self.R + torch.matmul(self.B.T, torch.matmul(P, self.B))
        B = torch.matmul(self.B.T, torch.matmul(P, self.A))
        return torch.linalg.solve(A, B)