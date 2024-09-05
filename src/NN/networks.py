import torch
from torch import nn
from .d2l import d2l_Module

class Base_Net(d2l_Module):  #@save

    """Basic NN to be used for Pseudo Outcomes"""

    def __init__(self, lr, wd=0.1 , hidden_layer_list = [256]):

        super().__init__()
        self.save_hyperparameters()

        layer_list = []
        for hidden_vals in hidden_layer_list:

            layer_list.append(nn.LazyLinear(hidden_vals))
            layer_list.append(nn.ELU())
        layer_list.append(nn.LazyLinear(1))

        self.net = nn.Sequential(*layer_list)
    
    def forward(self, X):
        return self.net(X)
    
    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),weight_decay=0.1, lr=self.lr)
