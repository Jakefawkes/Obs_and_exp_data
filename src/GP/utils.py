import torch
import yaml
import pandas as pd


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

def custom_confidence_region(MultivariateNormal,alpha=1.96):
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