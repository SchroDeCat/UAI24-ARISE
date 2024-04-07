from typing import Any, Tuple
from torchmetrics import MeanSquaredLogError
from math import log, sqrt
import torch
import tqdm
import os
import gpytorch
import pickle


def find_nearest_points(test_x:torch.tensor, x:torch.tensor, dim:int=1)->int:
    """
    Finds the nearest points in a n*m tensor test_x to a 1*m tensor x.

    Input:
        @test_x (torch.Tensor): A n*m tensor of points.
        @x (torch.Tensor): A 1*m tensor of points.

    Returns:
        @nearest_point_idx (int): the indice of the nearest points in test_x to x.
    """

    if type(test_x) is not torch.Tensor:
        test_x = torch.tensor(test_x)
    if type(x) is not torch.Tensor:
        x = torch.tensor(x)
    # Compute the distances between each point in test_x and x.
    if dim is None:
        distances = torch.sum((test_x.reshape([ -1]) - x.reshape([x.size(0), -1]))**2, dim=-1)
    else:
        distances = torch.sum((test_x - x)**2, dim=dim)

    # Find the indices of the nearest points in test_x to x.
    nearest_point_idx = torch.argmin(distances, dim=0)

    return nearest_point_idx

def posterior(model:Any, test_x: torch.tensor) -> Tuple[torch.tensor]:
    '''
    Generate posterior on given GP and list of candidates
    Input:
        @model: given GP
        @test_x: candidates to evaluate marginal posterior
    Return:
        @mean: posterior mean on test_x
        @variance: posterior variance on test_x
        @ucb: posterior ucb on test_x
        @lcb: posterior lcb on test_x
    '''
    model.model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        preds = model.model(test_x)
    return preds.mean, preds.variance, preds.confidence_region()[1], preds.confidence_region()[0]

def train_seq_nn(model:torch.nn.Module, train_x:torch.tensor, train_y: torch.tensor, **kwargs: Any)->torch.nn.Sequential:
    """
    Train a sequential neural network
    Input:
        @model: given neural network architecture
        @train_x: train set input
        @train_y: train set label
    Return:
        @model: trained model
    """
    verbose = kwargs.get('verbose', False)
    loss_type = kwargs.get('loss_type', 'mse')
    train_iter = kwargs.get('train_iter', 10)
    lr = kwargs.get('learning_rate', 1e-6)
    data_size = train_x.size(0)
    batch_size = kwargs.get('batch_size', data_size)

    dataloader = torch.utils.data.DataLoader(list(zip(train_x, train_y.reshape([data_size, 1]))), batch_size=batch_size, shuffle=True)

    model.train()

    # optimizer = torch.optim.Adam(params=model.arc.parameters(), lr=lr)
    optimizer = torch.optim.SGD(params=model.arc.parameters(), lr=lr)
    
    # "Loss"
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
    else:
        mps_device = torch.device('cpu')
    mps_device = torch.device('cpu')
    loss_type = loss_type.lower()
    if loss_type == "mse":
        loss_func = torch.nn.MSELoss().to(device=mps_device)
    elif loss_type == 'mlse':
        loss_func = MeanSquaredLogError().to(device=mps_device)
    elif loss_type== 'l1':
        loss_func = torch.nn.L1Loss().to(device=mps_device)
    else:
        raise NotImplementedError(f"{loss_type} not implemented")
    
    # training iterations
    iterator = tqdm.auto.tqdm(range(train_iter)) if verbose else range(train_iter)
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        loss_sum = 0
        for _x, _y in dataloader:
            # Get output from model
            output = model(_x)
            # Calc loss and backprop derivatives
            loss = loss_func(_y, output)
            loss.backward()
            optimizer.step()
            loss_sum += loss

        if verbose:
            iterator.set_postfix({"Loss": loss_sum.item() * batch_size / data_size})
    
    model.eval()
    return model

def load_from_pickle(file_path:str, task_name:str, method_name:str)->Any:
    '''
    Load data from pickle file
    Input:
        @file_path: file path to load
        @task_name: task name
        @method_name: method name
    Return:
        @data: loaded data
    '''
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    assert task_name in data.keys()
    assert method_name in data[task_name].keys()
    return data[task_name][method_name]

def real_beta(util_num:int, search_space:int, delta:float, horizon:int)->float:
    '''
    Compute the sqrt(beta) according to theory
    Input:
        @util_num: number of utility functions
        @search_space: train set input
        @delta: confidence level, Pr(out of bound) < delta
        @horizon: total number of iteration
    Return:
        @beta: 2 log (|util_num| * |search_space| * 1/horizon / delta)
    '''
    beta = sqrt(2 * log(util_num * search_space * delta * 1 * horizon / delta))
    return beta

def check_resutls_dir(folder_path = './results'):
    '''
    Check if the results folder exists. If not, create one.
    '''
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If not, create the folder
        os.makedirs(folder_path)
        print(f"The folder '{folder_path}' has been created.")
    else:
        print(f"The folder '{folder_path}' already exists.")