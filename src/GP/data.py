import torch
from collections import namedtuple
import gpytorch
from src.GP.GP import ExactGPModel
from sklearn.datasets import make_regression
import pandas as pd

kallus_func = lambda X,U,T,eps: 1+T+X+2*T*X+0.5*X**2+0.75*T*X**2+U+0.5*eps

cov_matricies = []
data = namedtuple("data","X T Y")

outcome_funcs = namedtuple("Outcome_funcs","cfounded_func uncfounded_func")

cfoundeded_func_plot = lambda X,T: 1+T+X[:,0]+2*T*X[:,0]+0.5*X[:,0]**2+0.75*T*X[:,0]**2+2*(T-0.5)*X[:,0]**2
uncfoundeded_func_plot = lambda X,T: 1+T+X[:,0]+2*T*X[:,0]+0.5*X[:,0]**2+0.75*T*X[:,0]**2

plot_outcome_funcs = outcome_funcs(cfounded_func=cfoundeded_func_plot,
              uncfounded_func=uncfoundeded_func_plot)

def return_CATE(outcome_func_tuple):
    CATE = lambda X: outcome_func_tuple.uncfounded_func(X,1) - outcome_func_tuple.uncfounded_func(X,0)
    return CATE

def return_CATE_GAP(outcome_func_tuple):
    TRUE_CATE = lambda X: outcome_func_tuple.uncfounded_func(X,1) - outcome_func_tuple.uncfounded_func(X,0)
    CFDED_CATE = lambda X: outcome_func_tuple.cfounded_func(X,1) - outcome_func_tuple.cfounded_func(X,0)
    CATE_GAP = lambda X: TRUE_CATE(X) - CFDED_CATE(X) 
    return CATE_GAP

def return_CATE_GAP_from_estimated(outcome_func_tuple,estimated_CATE):
    TRUE_CATE = lambda X: outcome_func_tuple.uncfounded_func(X,1) - outcome_func_tuple.uncfounded_func(X,0)
    CFDED_CATE = estimated_CATE
    CATE_GAP = lambda X: TRUE_CATE(X) - CFDED_CATE(X) 
    return CATE_GAP

def asses_fit(model,X_id,Y_id,X_od,Y_od):
    CATE_pred_guas_ID = model.CATE(X_id)
    CATE_pred_guas_OD = model.CATE(X_od)
    MSE_ID, COVERAGE_ID, Interval_width_ID = compare_cate_to_guas(Y_id,CATE_pred_guas_ID)
    MSE_OD, COVERAGE_OD, Interval_width_OD = compare_cate_to_guas(Y_od,CATE_pred_guas_OD)
    result_dict = {"MSE_ID":MSE_ID, "COVERAGE_ID":COVERAGE_ID, "Interval_width_ID":Interval_width_ID,
                   "MSE_OD":MSE_OD, "COVERAGE_OD":COVERAGE_OD, "Interval_width_OD":Interval_width_OD}
    # print(result_dict)
    return result_dict
    

def get_pseudo_outcome_data(exp_data,T_prop=0.5):
    pseudo_outcome = ((exp_data.T - T_prop)/((T_prop)*(1-T_prop))) * exp_data.Y  
    pseudo_data = data(X=exp_data.X, T=exp_data.T,Y=pseudo_outcome)
    return pseudo_data

def get_conditioned_data(initial_data,T_val):
    truth_vec = initial_data.T == T_val
    return data(X=initial_data.X[truth_vec],Y=initial_data.Y[truth_vec],T=initial_data.T[truth_vec])

def adjust_data(initial_data,adjustment_func):
    adjust_Y = initial_data.Y - adjustment_func(initial_data.X,initial_data.T)
    return data(X=initial_data.X,Y=adjust_Y,T=initial_data.T)

for i in range(2):
    t = i-0.5
    cov_matricies.append(torch.tensor([[1,t],[t,1]]))

def get_exp_sample_kallus(n_samples,X_range=(-1,1),T_prop=0.5,outcome_func = kallus_func):
    X = (X_range[1] - X_range[0]) * torch.rand(n_samples) + X_range[0]
    T = (torch.rand(n_samples) > (1-T_prop)).type(torch.float)
    U = torch.randn(n_samples)
    eps = torch.randn(n_samples)
    Y = outcome_func(X,U,T,eps)
    return data(X,T,Y)

def get_obs_sample_kallus(n_samples,X_range=(-3,3),T_prop=0.5,cov_matricies=cov_matricies,outcome_func = kallus_func):
    T = (torch.rand(n_samples) > (1-T_prop)).type(torch.float)
    X,U = torch.zeros(n_samples),torch.zeros(n_samples)
    for i in range(2):
        t_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc= torch.zeros(len(cov_matricies[i])) ,covariance_matrix= cov_matricies[i])
        sample = t_dist.sample((sum(T==i),))
        X[T==i],U[T==i] = sample[:,0],sample[:,1]
    eps = torch.randn(n_samples)
    Y = outcome_func(X,U,T,eps)
    return data(X,T,Y)

def get_gp_samples(X_eval,dimx=1,train_data=None,likelihood=None,kernel="RBF",num_samples_RFF = 100):

    if likelihood ==None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # if train_data == None:
    #     model = ExactGPModel(None, None, likelihood, dimx = hyperparam_dict["dimx"], lb=hyperparam_dict["lb"], ub=hyperparam_dict["ub"])
    # else:
    #    model =  ExactGPModel(train_data.X, train_data.Y, likelihood, dimx = hyperparam_dict["dimx"], lb=hyperparam_dict["lb"], ub=hyperparam_dict["ub"])
    #    model.train()
        
    model = ExactGPModel(None, None, likelihood, dimx =dimx,kernel=kernel,num_samples_RFF = num_samples_RFF)
    model.eval()
    return model(X_eval).sample()


dimx = 1
ub = 1.1*torch.ones(3)
ub[0] = 1.1
ub[-1] = 1.1
lb = 0.9*torch.ones(3)
lb[-1] = 0.9
loglikelihood0 = [None]
training_iterations = 2000
num_samples = 20
warmup_steps = 20
sqrbeta0 = 1.44

hyperparam_dict = {
    "dimx":dimx,
    "ub":ub,
    "lb":lb,
    "loglikelihood0":loglikelihood0,
    "training_iterations":training_iterations,
    "num_samples":num_samples,
    "warmup_steps":warmup_steps,
    "sqrbeta0":sqrbeta0

}

def RFF_GP(dimx,num_samples):
    rff_kern = gpytorch.kernels.RFFKernel(num_samples=num_samples,num_dims=dimx)
    weights = torch.randn(2*num_samples,requires_grad=False)
    def GP(X):
        z = rff_kern._featurize(x=X,normalize=True)
        return (z @ weights).squeeze()
    return GP
    
def GP_func(X_range,
               d=1,N_eval_points=5000,train_data=None,likelihood=None,scale=1,kernel="RBF",num_samples_RFF=100):
    if d==1: 
        X_eval = torch.linspace(X_range[0],X_range[1],steps=N_eval_points)
        Y_eval = get_gp_samples(X_eval,train_data=train_data,likelihood=likelihood,kernel=kernel,num_samples_RFF=num_samples_RFF)
        def GP(X):
            if len(X.shape) > 1:
                X.squeeze() 
            a = ((N_eval_points-1)*(X-X_range[0])/(X_range[1]-X_range[0])).int()
            return (scale*Y_eval[a]).squeeze()
        return GP
    if d>1 or kernel == "RFF":
        return RFF_GP(dimx=d,num_samples=num_samples_RFF)

def get_train_data_GP(generating_outcome_funcs,
                         n_samples_exp = 200,
                         n_samples_obs = 1000, 
                         exp_range = (-1,1),
                         obs_range = (-3,3),
                         d = 1,
                         T_prop = 0.5,
                         sigma_noise = 1,
                         kernel="RBF",
                         num_samples_RFF=100,
                         sklearn_gen = False
                         ):
    cfd_GPs = [0,0]
    ucfd_GPs = [0,0]

    for i in range(2):
        cfd_GPs[i] =GP_func(obs_range,kernel=kernel,num_samples_RFF=num_samples_RFF,d=d)

        ucfd_GPs[i] =GP_func(obs_range,train_data=None,scale=1,kernel=kernel,num_samples_RFF=num_samples_RFF,d=d)
    if sklearn_gen:
        X_obs,_ = make_regression(n_samples_obs,n_features=d,n_informative=1)
        X_exp,_ = make_regression(n_samples_exp,n_features=d,n_informative=1)
        X_obs,X_exp = torch.Tensor(X_obs),torch.Tensor(X_exp)

    else:
        X_range_obs = obs_range
        X_obs = (X_range_obs[1] - X_range_obs[0]) * torch.rand((n_samples_obs,d)) + X_range_obs[0]
        X_range_exp = exp_range
        X_exp = (X_range_exp[1] - X_range_exp[0]) * torch.rand((n_samples_exp,d)) + X_range_exp[0]

    # cfded_GP_func = lambda X,T: (1-T)*cfd_GPs[0](X) + (T)*cfd_GPs[1](X)
    cfded_GP_func = generating_outcome_funcs.cfounded_func
    ucfded_GP_func = lambda X,T: cfded_GP_func(X,T) + (1-T)*ucfd_GPs[0](X) + (T)*ucfd_GPs[1](X)
    outcome_funcs_GP = outcome_funcs(cfounded_func=cfded_GP_func,uncfounded_func=ucfded_GP_func)

    T_obs = (torch.rand(n_samples_obs) > (1-T_prop)).type(torch.float)
    Y_obs = cfded_GP_func(X_obs,T_obs) + sigma_noise*torch.randn(n_samples_obs)
    obs_data_GP = data(X=X_obs,Y=Y_obs,T=T_obs)


    T_exp = (torch.rand(n_samples_exp) > (1-T_prop)).type(torch.float)
    Y_exp = ucfded_GP_func(X_exp,T_exp) + sigma_noise*torch.randn(n_samples_exp)
    exp_data_GP = data(X=X_exp,Y=Y_exp,T=T_exp)

    return exp_data_GP,obs_data_GP,outcome_funcs_GP

def get_train_data_IHDP_Linear(n_samples_exp = 100,
                            T_prop = 0.5,
                            W_prop = 0.4,
                            WT_prop = 0.3,
                            sigma_noise = 0.5,
                            n_samples_obs = 1000,
                         num_samples_RFF=1500,
                         ):
   
   ihdp_table = pd.read_csv('src/data/ihdp.csv')
   ihdp_table.iloc[:,1:7] = (ihdp_table.iloc[:,1:7] - ihdp_table.iloc[:,1:7].mean())/(ihdp_table.iloc[:,1:7].std())
   
   df_obs = ihdp_table.sample(n=n_samples_obs,replace=True)
   T_obs = torch.tensor(df_obs.treat.values)
   X_obs = torch.tensor(df_obs.iloc[:,1:].values).type(torch.float)
   d = X_obs.shape[1]
   w0 = (torch.randn(d)* (torch.rand(d) >(1-W_prop))).type(torch.float)
   w1 = (torch.randn(d)* (torch.rand(d) >(1-WT_prop))).type(torch.float)
   cfd_GPs = [0,0]
   ucfd_GPs = [0,0]

   for i in range(2):
      cfd_GPs[i] =GP_func((-1,1),kernel="RFF",num_samples_RFF=1500,d=d)

      ucfd_GPs[i] =GP_func((-1,1),train_data=None,scale=1,kernel="RFF",num_samples_RFF=num_samples_RFF,d=d)

   cfded_GP_func = lambda X,T: (X @ w0) + T*(X @ w1)
   ucfded_GP_func = lambda X,T: cfded_GP_func(X,T) + (1-T)*ucfd_GPs[0](X) + (T)*ucfd_GPs[1](X)
   outcome_funcs_GP = outcome_funcs(cfounded_func=cfded_GP_func,uncfounded_func=ucfded_GP_func)

   Y_obs = cfded_GP_func(X_obs,T_obs) + sigma_noise*(torch.randn(len(T_obs)))

   df_exp = ihdp_table.sample(n=n_samples_exp,weights=((0.8*(ihdp_table.cig))* (0.8*(ihdp_table.sex))),replace=True)
   T_exp = (torch.rand(n_samples_exp) > (1-T_prop)).type(torch.float)
   X_exp = torch.tensor(df_exp.iloc[:,1:].values).type(torch.float)


   Y_exp = ucfded_GP_func(X_exp,T_exp) + sigma_noise*(torch.randn(len(T_exp)))

   obs_data_GP = data(X=X_obs,Y=Y_obs,T=T_obs)
   exp_data_GP = data(X=X_exp,Y=Y_exp,T=T_exp)

   return exp_data_GP,obs_data_GP,outcome_funcs_GP