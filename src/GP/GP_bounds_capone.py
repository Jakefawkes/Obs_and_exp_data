from copy import deepcopy
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gpytorch
from gpytorch.priors import UniformPrior
from matplotlib import pyplot as plt
from matplotlib import rc
from pyro.infer.mcmc import NUTS, MCMC
from gpytorch.priors import UniformPrior
import pyro


class ExactGPModel_CAPONE(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, dimx, lb, ub, kernel=False ):
        super(ExactGPModel_CAPONE, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if not kernel==False:
            if kernel._get_name()=='SpectralMixtureKernel':
                kernel.initialize_from_data(train_x, train_y)
        else:

            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dimx,
                                           lengthscale_prior=UniformPrior(lb[1:-1],
                                                                          ub[1:-1]),
                                           lengthscale_constraint=gpytorch.constraints.Interval(
                                               lb[1:-1], ub[1:-1])),
                outputscale_prior=UniformPrior(lb[0], ub[0]),
                outputscale_constraint=gpytorch.constraints.Interval(lb[0], ub[0])
            )
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
    def pred_CATE(self,x):

        n = x.shape[0]

        zeros_like_x = torch.full((x.shape[0],1), dtype=torch.long, fill_value=0)
        ones_like_x = torch.full((x.shape[0],1), dtype=torch.long, fill_value=1)

        out = self(torch.cat([x, x]),torch.cat([ones_like_x, zeros_like_x]))
        
        mean = out.mean.detach()[:n] - out.mean.detach()[n:]
        
        covar = out.lazy_covariance_matrix
        stdv = (covar[:n,:n].diag()+covar[n:,n:].diag() - 2*covar[n:,:n].diag()).detach()
        return mean, stdv 
    
    def bound_CATE(self,X_test,sqrbeta,add_vec=None):

        
        mean,stddev_bound = self.pred_CATE(X_test)
        
        if add_vec == None:
            add_vec = torch.zeros_like(mean)

        mean = mean + add_vec

        bounds = torch.cat([mean - sqrbeta * stddev_bound,torch.flip(mean + sqrbeta * stddev_bound,[0])])

        return bounds, mean, stddev_bound
    

def train(train_x, train_y, model0, likelihood0, n_training_iter):
    # Use the adam optimizer
    optimizer = torch.optim.SGD(model0.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll0 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood0, model0)
    for i in range(n_training_iter):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output0 = model0(train_x)
        # Calc loss and backprop derivatives
        loss = -mll0(output0, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f \r' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
        torch.cuda.empty_cache()
    return model0, likelihood0, mll0(output0, train_y)

def getboundinggp(sampmods, model0, nmc, delta_max):

    # set number of MCMC samples and delta if not available
    if not nmc:
        nmc = sampmods.covar_module.base_kernel.lengthscale.shape[0]
    if not delta_max:
        delta_max = 0.05

    # extract input dimension from lengthscales
    dimx = sampmods.covar_module.base_kernel.lengthscale.shape[-1]

    # concatenate hyperparameter samples generated with NUTS
    outputscale = sampmods.covar_module.outputscale  # corresponds to the signal variance (sigma_f^2)
    lengthscale = sampmods.covar_module.base_kernel.lengthscale  # lengthscale (NOT logarithm or square)
    noise = sampmods.likelihood.noise  # noise variance (\sigma_n^2)

    hyperparsamps = [[outputscale[i].reshape(1, 1), lengthscale[i].reshape(dimx, 1), noise[i].reshape(1, 1)]
                     for i in range(nmc)]
    hyperparsamps = [torch.cat(samps, 0).reshape(dimx + 2, ) for samps in hyperparsamps]

    outputscale0 = model0.covar_module.outputscale  # corresponds to the signal variance (sigma_f^2)
    lengthscale0 = model0.covar_module.base_kernel.lengthscale  # lengthscale (NOT logarithm or square)
    noise0 = model0.likelihood.noise  # noise variance (\sigma_n^2)

    theta0 = [outputscale0.reshape(1, 1), lengthscale0.reshape(dimx, 1), noise0.reshape(1, 1)]
    theta0 = torch.cat(theta0, 0).reshape(dimx + 2, )

    conf = 1 - delta_max
    dimpar = theta0.shape[0]

    indmax = round(len(hyperparsamps) * conf)
    inds = torch.as_tensor([torch.abs(samp - theta0).max() for samp in hyperparsamps]).argsort()[:indmax]
    sampsinregion = [hyperparsamps[ind] for ind in inds[:indmax]]
    thprim = torch.tensor([torch.tensor([samp[i] for samp in sampsinregion]).min() for i in range(dimpar)])
    thdoubprim = torch.tensor([torch.tensor([samp[i] for samp in sampsinregion]).max() for i in range(dimpar)])
    thprimnew = torch.min(thprim, theta0)
    thdoubprimnew = torch.max(thdoubprim, theta0)

    maxsqrbeta = 1.414
    gamma = torch.sqrt(torch.prod(torch.divide(thdoubprimnew[1:-1], thprimnew[1:-1])))
    gamma /= torch.sqrt(torch.prod(torch.divide(thdoubprimnew[0], theta0[0])))
    zeta = 0.1
    betabar = gamma ** 2 * (maxsqrbeta + zeta)
    beta = torch.as_tensor(min(4 * maxsqrbeta ** 2, betabar))
    sqrbeta = torch.sqrt(beta)

    # Create robust bounding hyperparameters. These correspond to the minimal lengthscales
    # and maximal signal/noise variances. Referred to as theta' in the experimental sectoin of the paper
    throbust = thprimnew  # deepcopy(thprimnew)
    throbust[0] = thdoubprimnew[0]  # deepcopy(thdoubprimnew[0])
    throbust[-1] = thdoubprimnew[-1]  # deepcopy(thdoubprimnew[-1])

    robustmodel = deepcopy(model0)
    robustmodel.covar_module.base_kernel._set_lengthscale(throbust[1:-1])
    robustmodel.covar_module._set_outputscale(throbust[0])
    robustmodel.likelihood.noise = throbust[-1]

    return robustmodel, sqrbeta, gamma

class BoundingGPModel_Unknown_Hyperparameters():

    def __init__(self,likelihood,hyperparam_dict) -> None:

        self.likelihood = likelihood

        self.dimx = hyperparam_dict.get("dimx")
        self.ub = hyperparam_dict.get("ub")
        self.lb = hyperparam_dict.get("lb")
        self.loglikelihood0 = hyperparam_dict.get("loglikelihood0")
        self.training_iterations = hyperparam_dict.get("training_iterations")
        self.num_samples = hyperparam_dict.get("num_samples")
        self.warmup_steps = hyperparam_dict.get("warmup_steps")
        self.sqrbeta0 = hyperparam_dict.get("sqrbeta0")

        self.model0 = None
        self.fullbmodel = None
        self.boundinggp = None

    def fit(self, X_train,Y_train):

        print("Training Base Model")
        likelihood0 = deepcopy(self.likelihood)

        model0 = ExactGPModel(X_train, Y_train, likelihood0, self.dimx, self.lb, self.ub)


        model0, likelihood0, self.loglikelihood0[0] = train(X_train, Y_train, model0, likelihood0, self.training_iterations)
        model0.eval()
        self.model0 = model0

        print("Training Bayes Model")

        blikelihood = deepcopy(likelihood0)
        fullbmodel = ExactGPModel(X_train, Y_train, blikelihood, self.dimx, self.lb, self.ub)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(blikelihood, fullbmodel)

        def pyro_model(x, y):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_fullbmodel = fullbmodel.pyro_sample_from_prior()
                output = sampled_fullbmodel.likelihood(sampled_fullbmodel(x))
                pyro.sample("obs", output, obs=y)
            return y


        nuts_kernel = NUTS(pyro_model)
        mcmc_run = MCMC(nuts_kernel, num_samples=self.num_samples, warmup_steps=self.warmup_steps, disable_progbar=False)
        print('Generating GP samples for fully Bayesian GP...')
        mcmc_run.run(X_train, Y_train)
        
        fullbmodel.pyro_load_from_samples(mcmc_run.get_samples())
        fullbmodel.eval()
        self.fullbmodel = fullbmodel

        boundinggp, sqrbetabar, gammaopt = getboundinggp(self.fullbmodel, self.model0, [], [])
        boundinggp.eval()
        self.boundinggp = boundinggp

        return None
    
    def bound_output(self,X_test):

        mean0 = self.model0(X_test).mean.detach()
        mean_bound, stddev_bound = self.boundinggp(X_test).mean.detach(), self.boundinggp(X_test).stddev.detach()

        bounds = torch.cat([mean0 - self.sqrbeta0 * stddev_bound,torch.flip(mean0 + self.sqrbeta0 * stddev_bound,[0])])

        return bounds, mean0, stddev_bound

    def add_bounds(self,X_test,mean_add, stddev_bound_add):

        mean0 = self.model0(X_test).mean.detach()
        mean_bound, stddev_bound = self.boundinggp(X_test).mean.detach(), self.boundinggp(X_test).stddev.detach()
        
        mean0 = mean0 + mean_add
        stddev_bound = (stddev_bound**2+stddev_bound_add**2)**0.5
        bounds = torch.cat([mean0 - self.sqrbeta0 * stddev_bound,torch.flip(mean0 + self.sqrbeta0 * stddev_bound,[0])])

        return bounds, mean0, stddev_bound