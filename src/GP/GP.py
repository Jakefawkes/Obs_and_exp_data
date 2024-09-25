from typing import Optional

import torch

from typing import Optional
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, dimx,kernel= "RBF",num_samples_RFF = 100):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if kernel =="RBF":
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dimx))
        if kernel =="RFF":
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RFFKernel(num_samples=num_samples_RFF,num_dims=dimx))
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class PseudoOutcome_StandardGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x_T, train_y, likelihood, p_score=0.5, kernel_list = [gpytorch.kernels.RBFKernel()]):
        super(PseudoOutcome_StandardGPModel, self).__init__(train_x_T, train_y,likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel_list[0]

    def forward(self,x,t):
        
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
    def CATE(self,x):
        t = torch.full((x.shape[0],1), dtype=torch.long, fill_value=2)
        return self(x,t)

class TrainedPseudoOutcome_MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x_T, train_y, likelihood, p_score=0.5, kernel_list =[gpytorch.kernels.RBFKernel(),gpytorch.kernels.RBFKernel()]):
        super(TrainedPseudoOutcome_MultitaskGPModel, self).__init__(train_x_T, train_y,likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.p_score = p_score
        self.covar_module_CATE = kernel_list[0]
        self.covar_module_nuisance = kernel_list[1]

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)

        self.task_covar_module_nuisance = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)
        self.task_covar_module_CATE = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self,x,t):
        
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x_CATE = self.covar_module_CATE(x)

        covar_t_CATE = self.task_covar_module_nuisance(t)

        covar_x_CATE = covar_x_CATE.mul(covar_t_CATE)
        
        covar_x_nuisance = self.covar_module_CATE(x)

        # Get task-task covariance
        covar_t_nuisance = self.task_covar_module_nuisance(t)
        # Multiply the two together to get the covariance we want
        covar_nuisance = covar_x_nuisance.mul(covar_t_nuisance)

        covar = covar_x_CATE + covar_nuisance
        # covar = covar_x_CATE 
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
    def CATE(self,x):

        t0 = torch.full((x.shape[0],1), dtype=torch.long, fill_value=0)
        t1 = torch.full((x.shape[0],1), dtype=torch.long, fill_value=1)

        n = x.shape[0]

        guas_cross = self(torch.cat([x,x]) ,torch.cat([t1,t0]))
        CATE_mean = self.p_score*guas_cross.mean[:n] + (1-self.p_score)*guas_cross.mean[n:]
        CATE_covar = (self.p_score**2)*guas_cross.covariance_matrix[:n,:n] + (1-self.p_score)**2*guas_cross.covariance_matrix[n:,n:] + (1-self.p_score)*self.p_score*(guas_cross.covariance_matrix[:n,n:] + guas_cross.covariance_matrix[n:,:n])
        CATE_covar = torch.diag(CATE_covar.diag())
        return gpytorch.distributions.MultivariateNormal(CATE_mean, CATE_covar)
    
class FixedPseudoOutcome_MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x_T, train_y, likelihood, p_score=0.5, kernel_list =[gpytorch.kernels.RBFKernel(),gpytorch.kernels.RBFKernel()],Train=False):
        super(FixedPseudoOutcome_MultitaskGPModel, self).__init__(train_x_T, train_y,likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        self.covar_module_CATE = kernel_list[0]
        self.covar_module_nuisance = kernel_list[1]

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)

        
        self.task_covar_module_nuisance = gpytorch.kernels.IndexKernel(num_tasks=3, rank=1)

        if Train:
            self.task_covar_module_nuisance.covar_factor = torch.nn.Parameter(torch.tensor([[-(1/(1-p_score))],[(1/p_score)],[0]]))
        
        else:
            self.task_covar_module_nuisance.covar_factor = torch.nn.Parameter(torch.tensor([[-(1/(1-p_score))],[(1/p_score)],[0]]),requires_grad=False)

        self.task_covar_module_nuisance.var = torch.tensor([0,0,0],requires_grad=False)

    def forward(self,x,t):
        
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x_CATE = self.covar_module_CATE(x)

        covar_x_nuisance = self.covar_module_nuisance(x)
        # Get task-task covariance
        covar_t_nuisance = self.task_covar_module_nuisance(t)
        # Multiply the two together to get the covariance we want
        covar_nuisance = covar_x_nuisance.mul(covar_t_nuisance)

        covar = covar_x_CATE + covar_nuisance
        # covar = covar_x_CATE 
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
    def CATE(self,x):

        t = torch.full((x.shape[0],1), dtype=torch.long, fill_value=2)
        return self(x,t)