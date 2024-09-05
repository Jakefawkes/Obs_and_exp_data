import torch
from .d2l import d2l_DataModule

def trial_data_generator(n):
    w=torch.tensor([2, -3.4])
    b=4.2
    noise=0.01
    X = torch.randn(n, len(w))
    noise = torch.randn(n, 1) * noise
    y = torch.matmul(X, w.reshape((-1, 1))) + b + noise
    t = torch.ones_like(y)
    return X,y,t,t,t

def get_IPW_psuedo_generator(sythetic_data_generator):
    
    """Gets IPW Pseudo Outcome Generator from a synthetic data generator
        Synethetic generator should ouput X, y, t, propsentity, CATE"""
    
    def IPW_psuedo_generator(n_samples):
        
        X, y, t, prop, cate = sythetic_data_generator(n_samples)
        
        weights = (t - prop)/(prop*(1-prop))

        return X, weights*y, cate, t
    
    return IPW_psuedo_generator

class Synthetic_data_loader(d2l_DataModule):

    def __init__(self,synethetic_generator, num_train=100, num_val=100,
                 batch_size=128,use_T=True):
        
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        X, self.y, t,self.prop,self.cate = synethetic_generator(n)

        
        if use_T:
            if len(t.shape) == 1:
                    t = t.unsqueeze(1)
                    self.X = torch.concat([X,t],dim=1)
                    self.t = t

        else:
            self.X = X

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                        shuffle=train)

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
    
class SyntheticPseudo_data_loader(d2l_DataModule):

    def __init__(self,synethetic_pseuedo_generator, num_train=100, num_val=100,
                 batch_size=128,use_T = False):
        
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        X, self.y, self.CATE,t = synethetic_pseuedo_generator(n)
        
        if use_T:
            if len(t.shape) == 1:
                t = t.unsqueeze(1)
                self.X = torch.concat([X,t],dim=1)
                self.t = t
        else: 
            self.X = X

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                        shuffle=train)

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)