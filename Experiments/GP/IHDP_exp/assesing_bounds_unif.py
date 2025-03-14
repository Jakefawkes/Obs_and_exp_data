"""
Description : Repeats the experiment from figure

Usage: assesing_bounds_unif.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
"""

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import sys
import os
import yaml
from docopt import docopt
import random
from tqdm import tqdm
from datetime import datetime
import warnings

running_dir = os.getcwd()
main_dir = running_dir.split("experiments")[0]
# Add the parent directory to the Python path
sys.path.append(main_dir)

from src.GP.data import *
from src.GP.plotting import *
from src.GP.GP import *
from src.GP.utils import *

args = docopt(__doc__)

# Load config file
with open(args['--cfg'], "r") as f:
    cfg = yaml.safe_load(f)

if cfg["experiment"]["surpress_warnings"]:
    warnings.filterwarnings("ignore")

GP_model_dict = {
    "fixed_MTGP":FixedPseudoOutcome_MultitaskGPModel,
    "standard_GP":PseudoOutcome_StandardGPModel,
    "trained_MTGP":TrainedPseudoOutcome_MultitaskGPModel
}

likelihood_dict = {
    "Guassian": gpytorch.likelihoods.GaussianLikelihood
}

kernel_dict = {
    "RBF": gpytorch.kernels.RBFKernel
}

if cfg["experiment"]["seed"] is not None:

    seed = cfg["experiment"]["seed"]
    random.seed(seed)
    torch.manual_seed(0)

T_prop = cfg["data"]["T_prop"]

tau = cfg["models"]["tau"]
results_dict = {}

# Create output directory if doesn't exists
now = datetime.now()
date_time_str = now.strftime("%m-%d %H:%M:%S")
direct_path = os.path.join(args['--o'],date_time_str.replace(" ","-"))
os.makedirs(direct_path, exist_ok=True)
with open(os.path.join(direct_path, 'cfg.yaml'), 'w') as f:
    yaml.dump(cfg, f)
dump_path = os.path.join(direct_path, 'results.metrics')

for model_name in cfg["models"]["model_list"]:
    results_dict[model_name] = {
        "MSE_ID" : [],
        "COVERAGE_ID": [],
        "Interval_width_ID": [],
        "MSE_OD" : [],
        "COVERAGE_OD": [],
        "Interval_width_OD": [],
    }

print("Starting")

if cfg["experiment"]["unif_bound"]:

    results_dict["Unif_COVERAGE_ID"] = []
    results_dict["Unif_Interval_width_ID"] = []

    results_dict["Unif_Interval_width_OD"] = []
    results_dict["Unif_COVERAGE_OD"] = []

for i in tqdm(range(cfg["experiment"]["n_runs"])):

    exp_data,obs_data,outcome_funcs_GP = get_train_data_IHDP_Linear_robust(
                            n_samples_exp = cfg["data"]["n_samples_exp"],
                            n_samples_obs = cfg["data"]["n_samples_obs"], 
                            T_prop = T_prop,
                            sigma_noise=cfg["data"]["sigma_noise"],
                            num_samples_RFF=cfg["data"]["num_samples_RFF"],
                            W_prop=cfg["data"]["W_prop"],
                            WT_prop=cfg["data"]["WT_prop"])
    
    cfoundeded_CATE_func = lambda X,T: outcome_funcs_GP.cfounded_func(X,1) - outcome_funcs_GP.cfounded_func(X,0)

    if cfg["models"]["fit_linear"]:
        XT_obs = torch.cat([obs_data.X,obs_data.X*obs_data.T.unsqueeze(1)],dim=1)
        beta =  (torch.inverse(XT_obs.T @ XT_obs) @ (XT_obs.T @ obs_data.Y))
        estimated_CATE_func = lambda X: torch.cat([torch.zeros_like(X),X],dim=1) @ beta

    else:
        estimated_CATE_func = cfoundeded_CATE_func
    adjustment_cate = lambda X,T : estimated_CATE_func(X)
    
    pseudo_data = get_pseudo_outcome_data(exp_data,T_prop=T_prop)
    pseudo_data_adjusted = adjust_data(pseudo_data,adjustment_cate)

    for model_name in cfg["models"]["model_list"]:

        likelihood = likelihood_dict[cfg["models"]["likelihood"]]()

        kernel_list = [
            kernel_dict[cfg["models"]["Kernel_0"]](ard_num_dims=28),
            kernel_dict[cfg["models"]["Kernel_1"]](ard_num_dims=28)
        ]
        model = GP_model_dict[model_name](
            train_x_T = (pseudo_data_adjusted.X,pseudo_data_adjusted.T),
            train_y = pseudo_data_adjusted.Y,
            likelihood = likelihood,
            p_score = T_prop,
            kernel_list = kernel_list
        )

        model.train()
        likelihood.train()
        if cfg["experiment"]["hyperparam_train"]:

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["models"]["lr"])  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(training_iterations):
                optimizer.zero_grad()
                output = model(pseudo_data_adjusted.X, pseudo_data_adjusted.T)
                loss = -mll(output, pseudo_data_adjusted.Y)
                loss.backward(retain_graph=True)
                optimizer.step()
        
        model.eval()
        likelihood.eval()
        exp_data_test,obs_data_test,_ = get_train_data_IHDP_Linear_robust(
                            n_samples_exp = 500,
                            n_samples_obs = 2000, 
                            T_prop = T_prop,
                            sigma_noise=cfg["data"]["sigma_noise"],
                            num_samples_RFF=cfg["data"]["num_samples_RFF"],
                            W_prop=cfg["data"]["W_prop"],
                            WT_prop=cfg["data"]["WT_prop"])
                                                                
        X_in_dist = exp_data_test.X
        X_out_dist = obs_data_test.X

        CATE_GAP_func = return_CATE_GAP_from_estimated(outcome_funcs_GP,estimated_CATE_func)

        x = X_in_dist
        CATE_GAP_ID = CATE_GAP_func(x)
        CATE_pred_guas_ID = model.CATE(x)
        MSE_ID, COVERAGE_ID, Interval_width_ID = compare_cate_to_guas(CATE_GAP_ID,CATE_pred_guas_ID)

        results_dict[model_name]["MSE_ID"].append(MSE_ID)
        results_dict[model_name]["COVERAGE_ID"].append(COVERAGE_ID)
        results_dict[model_name]["Interval_width_ID"].append(Interval_width_ID)
        
        x = X_out_dist
        CATE_GAP_OD = CATE_GAP_func(x)
        CATE_pred_guas_OD = model.CATE(x)

        MSE_OD, COVERAGE_OD, Interval_width_OD = compare_cate_to_guas(CATE_GAP_OD,CATE_pred_guas_OD)

        results_dict[model_name]["MSE_OD"].append(MSE_OD)
        results_dict[model_name]["COVERAGE_OD"].append(COVERAGE_OD)
        results_dict[model_name]["Interval_width_OD"].append(Interval_width_OD)

        if cfg["experiment"]["unif_bound"]:
            
            UNIF_COVERAGE_ID, UNIF_Interval_width_ID =compare_unif_bound_guas(CATE_GAP_ID,model,CATE_pred_guas_ID,tau= tau)
            UNIF_COVERAGE_OD, UNIF_Interval_width_OD =compare_unif_bound_guas(CATE_GAP_OD,model,CATE_pred_guas_OD,tau= tau)

            results_dict["Unif_COVERAGE_ID"].append(UNIF_COVERAGE_ID)
            results_dict["Unif_COVERAGE_OD"].append(UNIF_COVERAGE_OD)

            results_dict["Unif_Interval_width_ID"].append(UNIF_Interval_width_ID)
            results_dict["Unif_Interval_width_OD"].append(UNIF_Interval_width_OD)


    with open(dump_path, 'w') as f:
            yaml.dump(results_dict, f)


