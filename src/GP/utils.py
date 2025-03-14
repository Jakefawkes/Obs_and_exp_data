import torch
import yaml
import pandas as pd
import math

def calc_bound_IDHP(tau,alpha = 0.95):

    bounding_number = (1+1/tau)**7*2**21

    log_bounding_number = (2*math.log(bounding_number/alpha))**0.5

    return log_bounding_number,bounding_number

def calc_bounding_term(model,CATE_pred_guas,tau= 0.00000001):
    term_1 = (calc_bound_IDHP(tau)[0]*CATE_pred_guas.stddev)
    N = model.prediction_strategy.mean_cache.shape[0]
    term_2 = ((N)**0.5*model.prediction_strategy.mean_cache.norm()).item()*tau
    term_3 = calc_bound_IDHP(tau)[0]* (2*tau*(1+N*model.prediction_strategy.covar_cache.norm()))
    return term_1 + term_2 + term_3

def compare_unif_bound_guas(CATE,model,CATE_pred_guas,tau= 0.00000001):
    bounding_term = calc_bounding_term(model,CATE_pred_guas,tau= tau)
    region_lower,region_upper = CATE_pred_guas.mean - bounding_term, CATE_pred_guas.mean + bounding_term
    CATE_coverage = (torch.logical_and((region_lower < CATE),(region_upper > CATE))).float().mean()
    average_interval_width = (region_upper - region_lower).mean()
    return CATE_coverage.item(),average_interval_width.item()

def asses_fit(model,X_id,Y_id,X_od,Y_od):
    CATE_pred_guas_ID = model.CATE(X_id)
    CATE_pred_guas_OD = model.CATE(X_od)
    MSE_ID, COVERAGE_ID, Interval_width_ID = compare_cate_to_guas(Y_id,CATE_pred_guas_ID)
    MSE_OD, COVERAGE_OD, Interval_width_OD = compare_cate_to_guas(Y_od,CATE_pred_guas_OD)
    result_dict = {"MSE_ID":MSE_ID, "COVERAGE_ID":COVERAGE_ID, "Interval_width_ID":Interval_width_ID,
                   "MSE_OD":MSE_OD, "COVERAGE_OD":COVERAGE_OD, "Interval_width_OD":Interval_width_OD}
    # print(result_dict)
    return result_dict
    
def multidim_linspace(start, end, steps, dtype=torch.float32):
    """
    Create a multi-dimensional tensor with evenly spaced values between start and end.
    
    Args:
    start (torch.Tensor or list): Starting values for each dimension
    end (torch.Tensor or list): Ending values for each dimension
    steps (int or list of ints): Number of steps for each dimension
    dtype (torch.dtype, optional): Data type of the output tensor. Default is torch.float32
    
    Returns:
    torch.Tensor: A multi-dimensional tensor with evenly spaced values
    """
    # Convert start and end to tensors if they're not already
    start = torch.tensor(start, dtype=dtype)
    end = torch.tensor(end, dtype=dtype)
    
    # If steps is an integer, use the same number of steps for all dimensions
    if isinstance(steps, int):
        steps = [steps] * len(start)
    
    # Create a list to hold the linspaces for each dimension
    linspaces = []
    for i in range(len(start)):
        linspaces.append(torch.linspace(start[i], end[i], steps[i], dtype=dtype))
    
    # Use meshgrid to create the multi-dimensional tensor
    grids = torch.meshgrid(*linspaces, indexing='ij')
    
    # Stack the grids to create the final tensor
    return torch.stack(grids)

def custom_confidence_region(MultivariateNormal,alpha=1.96,uniform_bound = False):
        """
        Returns alpha standard deviations above and below the mean.

        :return: Pair of tensors of size `... x N`, where N is the
            dimensionality of the random variable. The first (second) Tensor is the
            lower (upper) end of the confidence region.
        """
        std2 = MultivariateNormal.stddev.mul_(alpha)
        mean = MultivariateNormal.mean
        return mean.sub(std2), mean.add(std2)

def compare_cate_to_guas(CATE,CATE_pred_guas):

    CATE_MSE = torch.mean((CATE - CATE_pred_guas.mean)**2)
    region_lower,region_upper = custom_confidence_region(CATE_pred_guas)
    CATE_coverage = (torch.logical_and((region_lower < CATE),(region_upper > CATE))).float().mean()
    average_interval_width = (region_upper - region_lower).mean()
    
    return CATE_MSE.item(),CATE_coverage.item(),average_interval_width.item()


def summerise_results(results_path):

    with open(results_path, "r") as f:
        results_dict = yaml.safe_load(f) 

    df = pd.DataFrame()
    for model_name in results_dict:
        model_df = pd.DataFrame(results_dict[model_name])
        model_df["model"] = model_name
        df = pd.concat([df,model_df])
    stats = df.groupby(['model']).agg(['mean', 'std'])

    return stats

def asses_fit(model,X_id,Y_id,X_od,Y_od):
    CATE_pred_guas_ID = model.CATE(X_id)
    CATE_pred_guas_OD = model.CATE(X_od)
    MSE_ID, COVERAGE_ID, Interval_width_ID = compare_cate_to_guas(Y_id,CATE_pred_guas_ID)
    MSE_OD, COVERAGE_OD, Interval_width_OD = compare_cate_to_guas(Y_od,CATE_pred_guas_OD)
    result_dict = {"MSE_ID":MSE_ID, "COVERAGE_ID":COVERAGE_ID, "Interval_width_ID":Interval_width_ID,
                   "MSE_OD":MSE_OD, "COVERAGE_OD":COVERAGE_OD, "Interval_width_OD":Interval_width_OD}
    # print(result_dict)
    return result_dict