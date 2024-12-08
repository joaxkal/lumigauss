import torch
from torch import nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, sh_degree=3, mlp_W=64, mlp_D=3, N_a=24):
        """
        """
        super().__init__()

        self.D = mlp_D -1
        self.W = mlp_W
        self.N_a = N_a

        self.sh_degree=sh_degree
        self.features_dc_dim = 1
        if self.sh_degree==0:
            self.features_rest_dim=0
        elif self.sh_degree==1:
            self.features_rest_dim=3
        elif self.sh_degree==2:
            self.features_rest_dim=8
        elif self.sh_degree==3:
            self.features_rest_dim=15
        else:
            raise NotImplemented('sh>3 not implemented')
   
        self.inputs_dim = self.N_a

        # encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.inputs_dim, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"encoding_{i+1}", layer)

        self.env_sh = nn.Linear(self.W, 3*(self.features_dc_dim+self.features_rest_dim))

    def forward(self, x):
        """
        """
 
        input = x
        for i in range(self.D):
            input = getattr(self, f"encoding_{i+1}")(input)

        outputs = self.env_sh(input)
        return outputs.reshape(3,(self.features_dc_dim+self.features_rest_dim))