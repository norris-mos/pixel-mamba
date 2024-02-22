import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../', 'src/pixel/data/models/')))
print(sys.path)
from pixba import modelling_pixba, configuration_pixba

batch, length, dim = 2, 64, 16

config = configuration_pixba.PIXBAConfig()   

x = torch.randn(batch, length, dim).to("cuda")
model = modelling_pixba.PIXBAModel(
    # This module uses roughly 3 * expand * d_model^2 parameters
    #d_model=dim, # Model dimension d_model
    #d_state=16,  # SSM state expansion factor
    #d_conv=4,    # Local convolution width
    #expand=2,    # Block expansion factor
    config
)
y = model(x)
print('working')
assert y.shape == x.shape
