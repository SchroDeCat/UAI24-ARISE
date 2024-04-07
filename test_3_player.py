'''
Test various acquisition functions on the games.
'''

import numpy as np
import warnings
import torch
import gpytorch
import tqdm
from matplotlib import pyplot as plt
from src import BO_NE, check_resutls_dir, RPS, hotelling, Saddle, BudgetAllocation, Saddle_var, hotelling3
from argparse import ArgumentParser
import cProfile, pstats, io
import datetime
import pickle
import random

TASKS = ['RPS', 'saddle', 'hotelling','BudgetAllocation'] # the regret for budget allocation is problematic
TASKS = ['hotelling', 'BudgetAllocation']
ACQS = ['ballet', 'pred', 'sur', 'epsilon_greedy']
SUBSAMPLE_NUM = 2000
TASK_SAMPLE_NUM = {
    'RPS': SUBSAMPLE_NUM,
    'saddle': 1000,
    'hotelling': SUBSAMPLE_NUM,
    'budgetallocation': 1000,
}

def main(task:str='RPS', acq_method:str='ts', retrain_interval:int=10, n_repeat:int=10, 
         opt_steps:int=10, train_iter:int=100, n_init:int=10, lr:float=1e-4, interpolate:bool=False, 
         verbose:bool=False, subsample:bool=False, spectral_norm:bool=False, gp_type:str='dk', beta:float=1,
         **kwargs)->np.ndarray:
    '''
    Inputs:
        @task: str, the task to run, one of ['RPS', 'hotelling', 'saddle']
        @acq_method: str, the acquisition function to use, one of ['random', 'ucb', 'ts', 'pred', 'sur', 'epsilon_greedy']
        @retrain_interval: int, the interval to retrain the model
        @n_repeat: int, the number of times to repeat the experiment
        @opt_steps: int, the number of optimization steps
        @train_iter: int, the number of training iterations
        @n_init: int, the number of initial points
        @lr: float, the learning rate
        @interpolate: bool, whether to use interpolation
        @verbose: bool, whether to print verbose information
        @subsample: bool, whether to subsample the test data
        @spectral_norm: bool, whether to use spectral normalization
        @gp_type: str, the type of GP to use, one of ['exact_gp', 'dk']
        @beta: float, the beta parameter for the acquisition function
    Return:
        @regret_rep: np.ndarray, the regret of each repeat
    '''
    Pr = cProfile.Profile()
    Pr.enable()

    if task.lower() == 'hotelling':
        problem = hotelling3()
    elif task.lower() == 'budgetallocation':
        problem = BudgetAllocation(n_agents=3)
    else:
        raise NotImplementedError(f"Task {task} not implemented")
    
    action_dim = problem.dim
    
    # Generate training labels, each label is a tuple of two floats that represent the utility for agent0 (i.e., u0(x0, x1) ) and agent1 (i.e., u1(x0, x1) ).
    train_x = problem.train_x
    _train_x = np.array(train_x).reshape([train_x.shape[0], -1])

    init_x = torch.from_numpy(_train_x[:n_init])
    init_y = np.array([problem.query_utilities(x.numpy()) for x in init_x])
    init_y_list = [torch.from_numpy(init_y[:, 0]), torch.from_numpy(init_y[:, 1]), torch.from_numpy(init_y[:, 2])]
    init_regret = np.array([problem.query_regrets(x.numpy()) for x in init_x])
    test_x = torch.from_numpy(np.copy(_train_x))


    n_rep = n_repeat
    fix_seed = True
    # gp_type = 'exact_gp'
    regret_rep = np.zeros([n_rep, opt_steps])
    regret = np.zeros(opt_steps)


    for rep in tqdm.auto.tqdm(range(n_rep), desc=f'method {acq_method}'):
        if fix_seed:
            _seed = init_x.size(0) + opt_steps * rep
            torch.manual_seed(_seed)
            np.random.seed(_seed)
            random.seed(_seed)
            torch.cuda.manual_seed(_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        iterator = tqdm.tqdm(range(opt_steps))
        for idx in iterator:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if idx >= 1:
                    pretrained_models = models
                    retrain=False if idx % retrain_interval != 0 else True
                else:
                    pretrained_models = None
                    retrain=True
                if acq_method.lower() == 'random':
                    retrain=False
                bo = BO_NE(
                    init_x,
                    init_y_list,
                    interpolation=interpolate,
                    gp_type=gp_type,
                    action_dim=action_dim,
                    train_iter=train_iter,   # default as 10.
                    lr=lr, 
                    pretrained_models=pretrained_models,
                    retrain=retrain,
                    verbose=verbose,
                    spectral_norm=spectral_norm,
                    noise_constraint=gpytorch.constraints.Interval(2e-4, 2.5e-3),
                )
                if subsample:
                    candidate = bo.query(test_x, acq=acq_method, subsample_num=TASK_SAMPLE_NUM[task.lower()], beta=beta, **kwargs)
                else:
                    candidate = bo.query(test_x, acq=acq_method, beta=beta, **kwargs)
                # 
                models = bo._models

            _new_x = candidate.numpy()
            _new_y_1, _new_y_2, _new_y_3 = problem.query_utilities(_new_x)
            _tmp_x = torch.from_numpy(_new_x).unsqueeze(dim=0)
            init_x = torch.cat([init_x, _tmp_x], dim=0)
            init_y_list[0] = torch.cat([init_y_list[0], torch.from_numpy(_new_y_1.reshape(1))], dim=0)
            init_y_list[1] = torch.cat([init_y_list[1], torch.from_numpy(_new_y_2.reshape(1))], dim=0)
            init_y_list[2] = torch.cat([init_y_list[2], torch.from_numpy(_new_y_3.reshape(1))], dim=0)
            regret[idx] = problem.query_regrets(_new_x)
            regret[:idx+1] = np.minimum.accumulate(regret[:idx+1])
            if verbose and acq_method.lower() == 'ballet':
                _roi_test_x = test_x[bo._roi_filter]
                _roi_regret = np.array([problem.query_regrets(x) for x in _roi_test_x])
                iterator.set_postfix_str(f"regret {regret[idx]:.2e} init {init_regret.min():.2e} roi_size {bo._roi_filter.sum():.1e}/{test_x.size(0):.1e} roi_regret ({_roi_regret.min():.2e},{_roi_regret.max():.2e}) lcb ({bo._lcb_f.min():.2e}, {bo._lcb_f.max():.2e}) ucb {bo._ucb_f.min():.2e} beta={bo.beta:.2e}")    
            else:
                iterator.set_postfix_str(f"regret {regret[idx]:.2e}")
                if regret[idx] < 0.1:
                    iterator.set_postfix_str(f"regret {regret[idx]:.2e}  candidate {candidate}")

                    
        regret_rep[rep] = regret

    Pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(Pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    with open("profile.txt", "w") as f:
        f.write(s.getvalue())

    return regret_rep

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--name', type=str, default='debug')
    args.add_argument('--n_repeat', type=int, default=1)
    args.add_argument('--opt_steps', type=int, default=10)
    args.add_argument('--early_save',action='store_true')
    args.add_argument('--retrain_interval', type=int, default=10)
    args.add_argument('--interpolate', action='store_true')
    args.add_argument('--verbose', action='store_true')
    args.add_argument('--spectrum_norm', action='store_true')
    args.add_argument('--subsample', action='store_true')
    args.add_argument('--train_iter', type=int, default=10)
    args.add_argument('--n_init', type=int, default=10)
    args.add_argument('--gp_type', type=str, default='dk')
    args.add_argument('--task', type=str, default='hotelling')
    # args.add_argument('--acq_method', type=str, default='ts')
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--beta', type=float, default=2)
    args.add_argument('--dir', type=str, default='./results/uai_rebuttal/debug')
    args.add_argument('--global_search', action='store_true') # otherwise, by default BALLET (ARISE) is local
    args = args.parse_args()


    # parse arguments
    n_init = args.n_init
    subsample = args.subsample
    n_repeat = args.n_repeat
    opt_steps = args.opt_steps
    retrain_interval = args.retrain_interval
    interpolate = args.interpolate
    train_iter = args.train_iter
    gp_type = args.gp_type
    beta = args.beta
    on_roi = not args.global_search
    lr = args.lr
    regret_dict = {}
    check_resutls_dir(args.dir)
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    # traverse all the acquisition functions and dump into dict
    # for task in ['BudgetAllocation', 'hotelling']:
    assert args.task in ['BudgetAllocation', 'hotelling'], f"Task(3P) {args.task} not implemented"
    for task in [args.task]:
        target_dir = f"{args.dir}/{task.upper()}_3P_{gp_type}_SimReg_B{beta}_R{n_repeat}_OS{opt_steps}_TI{train_iter}_RI{retrain_interval}_NI{args.n_init}{'_IP' if interpolate else ''}{'_SS' if subsample else''}_{time_stamp}_{args.name}{'_roi' if on_roi else ''}.p"

        fig = plt.figure(figsize=(10, 10))
        plt.xlabel('Iteration')
        plt.ylabel('Simple Regret')
        plt.title(f'{task.upper()}')
        regret_dict[task] = {}
        for acq_method in ACQS:
            print(f"{'='*100}\n \t Start task {task} with method {acq_method} \n{'='*100}")
            regret = main(task=task, acq_method=f'{acq_method}', train_iter=train_iter, 
                          retrain_interval=retrain_interval, opt_steps=opt_steps, n_repeat=n_repeat, 
                          lr=lr, interpolate=interpolate, n_init=n_init, verbose=args.verbose, 
                          subsample=subsample, gp_type=gp_type, spectral_norm=args.spectrum_norm, beta=beta, 
                          on_roi=on_roi)   
            regret_dict[task][acq_method] = regret
            plt.plot(np.mean(regret, axis=0), label=acq_method)
            # multiple saves
            plt.legend()
            if args.early_save:
                pickle.dump(regret_dict, open(target_dir, "wb"))
                plt.savefig(f"{target_dir}ng")
            print(f"{'='*100}\n \t Finish task {task} with method {acq_method} \n{'='*100}")

        # dump the regret into a pickle file with current time in the file name
        pickle.dump(regret_dict, open(target_dir, "wb"))
        print(f'File stored:\n {target_dir}')
        print(f"{'-'*100}\n")
        plt.savefig(f"{target_dir}ng")
        plt.close()
