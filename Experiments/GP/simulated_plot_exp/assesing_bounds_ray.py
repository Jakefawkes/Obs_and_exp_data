"""
Description : Repeats the experiment from figure

Usage: assesing_bounds.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
"""


import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import sys
import yaml
from docopt import docopt
import random
from tqdm import tqdm
from datetime import datetime
import warnings
import time
import ray

ray.shutdown()
ray.init(ignore_reinit_error=True);

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

outcome_func_dict = {
    "plot_outcome_funcs":plot_outcome_funcs
}

experiment_outcome_funcs = outcome_func_dict[cfg["data"]["outcome_funcs"]]

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

cfoundeded_CATE_func = lambda X,T: experiment_outcome_funcs.cfounded_func(X,1) - experiment_outcome_funcs.cfounded_func(X,0)

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

@ray.remote(num_cpus=4)
def run_exp(
    cfg, 
    plot_outcome_funcs, 
    cfoundeded_CATE_func, 
    GP_model_dict, 
    likelihood_dict, 
    kernel_dict, 
):
    res_dict = {}
    for model_name in cfg["models"]["model_list"]:
        res_dict[model_name] = {
            "MSE_ID" : 0,
            "COVERAGE_ID": 0,
            "Interval_width_ID": 0,
            "MSE_OD" : 0,
            "COVERAGE_OD": 0,
            "Interval_width_OD": 0,
        }
    T_prop = cfg["data"]["T_prop"]

    exp_data,_,outcome_funcs_GP = get_train_data_GP(
        plot_outcome_funcs,
        n_samples_exp = cfg["data"]["n_samples_exp"],
        n_samples_obs = cfg["data"]["n_samples_obs"], 
        exp_range = cfg["data"]["exp_range"],
        obs_range = cfg["data"]["obs_range"],
        T_prop = T_prop,
        sigma_noise=cfg["data"]["sigma_noise"],
        kernel=cfg["data"]["data_generating_kernel"],
        num_samples_RFF=cfg["data"]["num_samples_RFF"],
        d=cfg["data"]["d"]
    )
        
    pseudo_data = get_pseudo_outcome_data(exp_data,T_prop=T_prop)
    pseudo_data_adjusted = adjust_data(pseudo_data,cfoundeded_CATE_func)

    for model_name in cfg["models"]["model_list"]:

        likelihood = likelihood_dict[cfg["models"]["likelihood"]]()

        kernel_list = [
            kernel_dict[cfg["models"]["Kernel_0"]](ard_num_dims=cfg["data"]["d"]),
            kernel_dict[cfg["models"]["Kernel_1"]](ard_num_dims=cfg["data"]["d"])
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

        CATE_GAP_func = return_CATE_GAP(outcome_funcs_GP)

        X_range_exp = cfg["data"]["exp_range"]

        X_range_obs = cfg["data"]["obs_range"]

        X_in_dist = (X_range_exp[1] - X_range_exp[0]) * torch.rand((cfg["experiment"]["n_evaluation_samples"],cfg["data"]["d"])) + X_range_exp[0]
                    
        x = X_in_dist
        CATE_GAP_ID = CATE_GAP_func(x)
        CATE_pred_guas_ID = model.CATE(x)
        MSE_ID, COVERAGE_ID, Interval_width_ID = compare_cate_to_guas(CATE_GAP_ID,CATE_pred_guas_ID)

        res_dict[model_name]["MSE_ID"] = MSE_ID
        res_dict[model_name]["COVERAGE_ID"] = COVERAGE_ID
        res_dict[model_name]["Interval_width_ID"] = Interval_width_ID
        
        X_out_dist = (X_range_exp[0] - X_range_obs[0]) * torch.rand(((int(cfg["experiment"]["n_evaluation_samples"]/2)),cfg["data"]["d"])) + X_range_exp[1]

        X_out_dist = torch.cat([X_out_dist,
                                (X_range_obs[1] - X_range_exp[1]) * torch.rand(((int(cfg["experiment"]["n_evaluation_samples"]/2)),cfg["data"]["d"])) + X_range_exp[1]],dim=0)

        x = X_out_dist
        CATE_GAP_OD = CATE_GAP_func(x)
        CATE_pred_guas_OD = model.CATE(x)
        MSE_OD, COVERAGE_OD, Interval_width_OD = compare_cate_to_guas(CATE_GAP_OD,CATE_pred_guas_OD)

        res_dict[model_name]["MSE_OD"] = MSE_OD
        res_dict[model_name]["COVERAGE_OD"] = COVERAGE_OD
        res_dict[model_name]["Interval_width_OD"] = Interval_width_OD

        return res_dict


cfg_id = ray.put(cfg)

futures = [run_exp.remote(
    cfg_id, 
    plot_outcome_funcs, 
    cfoundeded_CATE_func, 
    GP_model_dict, 
    likelihood_dict, 
    kernel_dict, 
) for _ in range(cfg["experiment"]["n_runs"])]

start = time.time()

while len(futures):
    ready, futures = ray.wait(futures)
    res = ray.get(ready[0])
    for model_name in res:
        results_dict[model_name]["MSE_ID"].append(res[model_name]["MSE_ID"])
        results_dict[model_name]["COVERAGE_ID"].append(res[model_name]["COVERAGE_ID"])
        results_dict[model_name]["Interval_width_ID"].append(res[model_name]["Interval_width_ID"])
        results_dict[model_name]["MSE_OD"].append(res[model_name]["MSE_OD"])
        results_dict[model_name]["COVERAGE_OD"].append(res[model_name]["COVERAGE_OD"])  
        results_dict[model_name]["Interval_width_OD"].append(res[model_name]["Interval_width_OD"])
    n_finished = cfg["experiment"]["n_runs"] - len(futures)
    if n_finished % 10 == 0:
        time_per_run = (time.time() - start) / n_finished
        print(f"Finished {n_finished} experiments. Time per experiment: {time_per_run:.2f}s")


with open(dump_path, 'w') as f:
    yaml.dump(results_dict, f)


