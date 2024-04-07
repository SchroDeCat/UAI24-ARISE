import gpytorch
import torch
import tqdm
import numpy as np

from scipy.interpolate import RBFInterpolator
from .sgld import SGLD
from typing import Any, List
from collections.abc import Iterable
from sklearn.preprocessing import RobustScaler, StandardScaler
from ..model import GPRegressionModel, ExactGPRegressionModel, RFFGPRegressionModel, LargeFeatureExtractor
from ..utils import find_nearest_points, real_beta
DEVICE = "cpu"
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPE = torch.float

class BO_NE():
    """
    BO for Nash Quilibrium Pipeline. Note the objective is to be minimized
    """
    def __init__(self, 
                init_x:torch.tensor,
                init_y_list:List[torch.tensor], 
                action_dim:int=1,
                lr:float=1e-6, 
                train_iter:int=10, 
                spectrum_norm:bool=False,
                verbose:bool=False, 
                robust_scaling:bool=True, 
                low_dim:bool=True,
                gp_type:str='exact_gp',
                loss_type:str='nll',
                noise_constraint:gpytorch.constraints.Interval=None, 
                output_scale_constraints:gpytorch.constraints.Interval=None,
                interpolation:bool=True,
                cuda:bool=False,
                **kwargs:Any,
                )->None:
        # scale input
        ScalerClass = RobustScaler if robust_scaling else StandardScaler
        self.scaler = ScalerClass().fit(init_x)
        init_x = torch.from_numpy(self.scaler.transform(init_x))
        # init vars
        self.lr = lr
        self.low_dim = low_dim
        self.verbose = verbose
        self.init_x = init_x
        self.init_y_list = [init_y.clone() for init_y in init_y_list]
        self.n_init = init_x.size(0)
        self.data_dim = init_x.size(1)
        self.interpolation = interpolation
        self.action_size = action_dim
        assert self.init_x.size(-1) % self.action_size == 0
        self.cuda = cuda
        self.model_num = len(init_y_list)
        for init_y in init_y_list:
            assert init_x.size(0) == init_y.size(0)
        self.train_iter = train_iter

        self.spectrum_norm = spectrum_norm
        self.noise_constraint = noise_constraint
        self.scale_constraints = output_scale_constraints
        self.gp_type = gp_type.lower()
        self.loss_type = loss_type.lower()

        # self.cuda = torch.cuda.is_available()
        if self.interpolation: # prior with interpolator
            self.interpolation_y_list = [None for _ in range(self.model_num)]
            interpolate_y = init_x.clone()
            interpolate_d_list = [init_y.clone().reshape([self.init_x.size(0)]) for init_y in init_y_list]
            _kernel = 'linear'
            self.interpolator_list = [RBFInterpolator(y=interpolate_y, d=interpolate_d, smoothing=1e-4, kernel=_kernel) for interpolate_d in interpolate_d_list]
            for idx in range(self.model_num):
                self.interpolation_y_list[idx] = self.interpolator_list[idx](interpolate_y).squeeze()
                self.init_y_list[idx] = self.init_y_list[idx] - self.interpolation_y_list[idx]

        if self.cuda:
            self.train_x = self.init_x.to(device=DEVICE, dtype=TYPE)
            self.train_y_list = [init_y.to(device=DEVICE, dtype=TYPE) for init_y in self.init_y_list]
        else:
            self.train_x = self.init_x.float().clone()
            self.train_y_list = [init_y.float().clone() for init_y in self.init_y_list]

        
        # init model
        self._likelihoods = [gpytorch.likelihoods.GaussianLikelihood(noise_constraint=self.noise_constraint) for _ in range(self.model_num)]
        self.feature_extractor = LargeFeatureExtractor(self.data_dim, self.low_dim, spectrum_norm=self.spectrum_norm)
        if  self.gp_type == 'exact_gp':
            self._models = [ExactGPRegressionModel(
                            train_x=self.train_x,
                            train_y=self.train_y_list[idx], 
                            gp_likelihood=self._likelihoods[idx],
                            low_dim=self.low_dim, 
                            output_scale_constraint=self.noise_constraint,
                            device=self.train_x.device,
                        ) for idx in range(self.model_num)]
        elif self.gp_type == 'dk':
            self._models = [GPRegressionModel(
                            train_x=self.train_x,
                            train_y=self.train_y_list[idx], 
                            gp_likelihood=self._likelihoods[idx],
                            gp_feature_extractor=self.feature_extractor, 
                            low_dim=self.low_dim, 
                            output_scale_constraint=self.noise_constraint,
                            device=self.train_x.device,
                        ) for idx in range(self.model_num)]
        elif self.gp_type == 'rff':
            self._models = [RFFGPRegressionModel(
                            train_x=self.train_x,
                            train_y=self.train_y_list[idx], 
                            gp_likelihood=self._likelihoods[idx],
                            low_dim=self.low_dim, 
                            output_scale_constraint=self.noise_constraint,
                            device=self.train_x.device,
                        ) for idx in range(self.model_num)]
        else:
            raise NotImplementedError(f"{self.gp_type} not implemented")

        pretrained_models = kwargs.get('pretrained_models', None)
        if pretrained_models:
            for model, pretrained_model in zip(self._models, pretrained_models):
                model.load_state_dict(pretrained_model.state_dict())
            pass

        self.models = gpytorch.models.IndependentModelList(*self._models)
        self.likelihoods = gpytorch.likelihoods.LikelihoodList(*self._likelihoods)
        

        retrain = kwargs.get('retrain', True)
        if retrain:
            self.train(verbose=self.verbose, loss_type=self.loss_type)

    def _interpolation_calibrate(self, test_x:torch.tensor, target_value:bool=None, cuda:bool=False, model_idx:int=0)->torch.Tensor:
        '''
        Calibrate the target value with interpolation on test_x if needed
        '''
        test_x = test_x.cpu()
        if self.interpolation:
            interpolator = self.interpolator_list[model_idx]
            _interpolation = torch.from_numpy(interpolator(test_x.cpu())).float()
            if cuda:
                _interpolation = _interpolation.cuda()
        else:
            _interpolation = 0
        
        if target_value is None:
            return _interpolation
        else:
            return target_value + _interpolation

    def _mll_loss(self, pred:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        assert hasattr(self, 'mll')
        #print(y[0].device, "debug"*100)
        return -self.mll(pred, y)

    def _mse_loss(self, pred:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        tmp_loss = torch.nn.MSELoss()
        return tmp_loss(pred.mean, y)
    
    def train(self, verbose:bool=False, loss_type:str='nll')->None:
        self.models.train()
        self.likelihoods.train()

        # optimizer
        if self.gp_type == 'dk' and self.spectrum_norm:
            self.optimizer = SGLD(self.models.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(params=self.models.parameters(), lr=self.lr)
        
        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihoods, self.models)
        if loss_type.lower() == "nll":
            self.loss_func = self._mll_loss
        elif loss_type.lower() == "mse":
            self.loss_func = self._mse_loss
        else:
            raise NotImplementedError(f"{loss_type} not implemented")
        
        # training iterations
        iterator = tqdm.auto.tqdm(range(self.train_iter)) if verbose else range(self.train_iter)
        for i in iterator:
            # Zero backprop gradients
            self.optimizer.zero_grad()
            # Get output from model
            self.output = self.models(*self.models.train_inputs)
            # Calc loss and backprop derivatives
            self.loss = self.loss_func(self.output, self.models.train_targets)#_on_device)
            self.loss.backward()
            self.optimizer.step()

            if verbose:
                iterator.set_postfix({"Loss": self.loss.item()})

    def optimize_action_set(self, test_x: torch.Tensor, y: torch.Tensor, agent_no:int=2, return_index:bool=False) -> torch.Tensor:
        '''
        Optimize action set for certain gp
        Input:
            @test_x: discrete search space
            #agent_no: number of agent to be fixed
            @y:  either prediciton or samples, could have multiple sample for a single point
        Return:
            maximum of y when tweaking the action of the certain agent while fixing the remaining of x
            Shape should be identical to y
        '''
        data_size = test_x.size(0)
        data_dim = test_x.size(-1)
        assert data_size == y.size(0)
        assert data_dim % self.action_size == 0

        optima = y.clone()                      # return optima
        remain = torch.ones(data_size) > 0      # all True
        util_array = torch.arange(start=0, end=data_size, dtype=torch.long)     # utility array
        optima_index = torch.zeros(y.shape).long()      # index of optima
        action_dim = torch.ones(data_dim)
        _agent_action_dim = self.action_size * agent_no
        action_dim[_agent_action_dim:_agent_action_dim + self.action_size] = 0  # inactive dims
        acting_x = test_x[:,action_dim==1]   # only the first action_dim is used
        while remain.any():
            _x_remain = acting_x[remain]
            _x = _x_remain[0]
            _x_filter = torch.linalg.norm(_x_remain - _x, ord=2, dim=-1) < 1e-10
            _change_nume = _x_filter.sum()
            assert _x_filter.any()
            _opt = optima[remain][_x_filter].max(dim=0)
            _opt_values = _opt.values
            _opt_indices = _opt.indices
            _tmp = optima[remain][_x_filter]
            _idcs = util_array[remain][_x_filter]
            optima[_idcs] =  _opt_values # update optima for a certain group
            optima_index[_idcs] = torch.tensor([util_array[remain][_x_filter][idx] for idx in _opt_indices.reshape([-1,1])]).long() # update the index of optima
            _tmp = optima[remain][_x_filter]
            remain[util_array[remain][_x_filter]] = False                    # update the remaining

        assert optima.size() == y.size()

        if return_index:
            assert torch.all(optima - y[optima_index] < 1e-10)
            return optima_index, optima
        else:
            return optima

    def ballet_query(self, test_x:torch.Tensor, minimum_choice:int=10, **kwargs)->torch.Tensor:
        '''
        ARISE: BO for Nash Equilibrium with ballet ROI-identification and mutli-model learning
        Input:
            @test_x: discrete search space
            @acq: acquisition function
            @method: sampling option
            @intersection: whether to use intersection of ROI
            @minimum_choice: minimum number of points to choose for ROI
        Output:
            candidate: to be evaluated (with minimum gap)
        '''
        # Set into eval mode
        self.models.eval()
        self.likelihoods.eval()

        if self.cuda:
            _test_x = test_x.to(device=DEVICE, type=TYPE)
        else:
            _test_x = test_x.float()
        
        # Identify ROI with minimum of UCB
        beta = kwargs.get('beta', 2)
        if beta == 0:
            beta = real_beta(self.model_num, self.init_x.size(0), delta=.1, horizon=100)
        assert beta > 0
        self.beta = beta
        self._ucb_f = torch.zeros(_test_x.size(0))
        self._lcb_f = torch.zeros(_test_x.size(0))
        # Actually LCB in the NE setting: we take the argmin
        for model_idx, (model, likelihood) in enumerate(zip(self.models.models, self.likelihoods.likelihoods)):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(_test_x))
                _std2 = observed_pred.stddev.mul_(beta) # default is 2
                _mean = observed_pred.mean
                lower, upper = _mean.sub(_std2), _mean.add(_std2)
                # lower, upper = observed_pred.confidence_region() # 2 standard 

            lcb = self._interpolation_calibrate(test_x=_test_x, target_value=lower, cuda=self.cuda, model_idx=model_idx)
            ucb = self._interpolation_calibrate(test_x=_test_x, target_value=upper, cuda=self.cuda, model_idx=model_idx)

            _partial_ucb = self.optimize_action_set(_test_x, ucb, model_idx, return_index=False)
            _partial_lcb = self.optimize_action_set(_test_x, lcb, model_idx, return_index=False)
            _partial_lcb_f = _partial_lcb - ucb 
            _partial_ucb_f = _partial_ucb - lcb
            self._ucb_f += _partial_ucb_f
            self._lcb_f += _partial_lcb_f
        self.acq_vals = self._ucb_f - self._lcb_f

        # filter ROI
        threshold = min(self._ucb_f.min(), 0)
        self._roi_filter = self._lcb_f <= threshold
        assert self._roi_filter.sum() <= _test_x.size(0)
        if self._roi_filter.sum() < minimum_choice:
            self._roi_filter = torch.argsort(self._lcb_f, descending=False)[:minimum_choice]
            self._roi_filter = torch.zeros(_test_x.size(0)).bool().scatter_(0, self._roi_filter, True)
        
        # maximize the acquisition function
        on_roi = kwargs.get('on_roi', True) # allow ablation study
        if on_roi:
            idx = self.acq_vals[self._roi_filter].argmax()
            idx = torch.arange(_test_x.size(0))[self._roi_filter][idx]
        else:
            idx = self.acq_vals.argmax()
        return _test_x[idx]       

    def query(self, test_x:torch.Tensor,  acq:str="ts", method:str="love", **kwargs)->torch.Tensor:
        '''
        BO for Nash Equilibrium
        Input:
            @test_x: discrete search space
            @acq: acquisition function
            @method: sampling option
        Output:
            candidate: to be evaluated (with minimum gap)
        '''
        if  not kwargs.get('no_scaling', False):
            test_x = torch.from_numpy(self.scaler.transform(test_x))

        if acq.lower() in ['ballet']:
            candidate = self.ballet_query(test_x=test_x, **kwargs)
            if not kwargs.get('no_scaling', False):
                candidate = torch.from_numpy(self.scaler.inverse_transform(candidate.unsqueeze(0)).squeeze())
                return candidate

        # Set into eval mode
        self.models.eval()
        self.likelihoods.eval()

        if self.cuda:
            _test_x = test_x.to(device=DEVICE, type=TYPE)
        else:
            _test_x = test_x.float()
        
        self.acq_vals = []
        _subsample = 'subsample_num' in kwargs
        if _subsample:
            _test_x_size = _test_x.size(0)
            _subsample_num = min(kwargs.get('subsample_num', 2000),  _test_x_size)
            assert _subsample_num <= _test_x_size
            _test_x = _test_x[np.random.choice(_test_x_size, _subsample_num)]


        if acq.lower() in ["ts", 'qei', 'pe']:
            # Either Thompson Sampling or Monte-Carlo EI
            _num_sample = 1 if acq.lower() == 'ts' else 100
            for model_idx, (model, likelihood) in enumerate(zip(self.models.models, self.likelihoods.likelihoods)):
                if method.lower() == "love":
                    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(200):
                        with gpytorch.settings.fast_pred_samples(): # NEW FLAG FOR SAMPLING
                                # model(_test_x)
                                samples = model(_test_x).rsample(torch.Size([_num_sample]))
                elif method.lower() == "ciq":
                    with torch.no_grad(), gpytorch.settings.ciq_samples(True), gpytorch.settings.num_contour_quadrature(10), gpytorch.settings.minres_tolerance(1e-4): 
                                samples = likelihood(model(_test_x)).rsample(torch.Size([_num_sample]))
                else:
                    raise NotImplementedError(f"sampling method {method} not implemented")
                
                samples = self._interpolation_calibrate(test_x=_test_x, target_value=samples, cuda=self.cuda, model_idx=model_idx)

                _maxima = self.optimize_action_set(_test_x, samples.T, model_idx)
                samples = _maxima.T - samples

                if acq.lower() == 'ts':
                    acq_val = samples.T.reshape([_test_x.shape[0], -1])
                elif acq.lower() == 'pe':
                    # probability of equilibrium, adopted from monte-carlo version from (https://arxiv.org/abs/1611.02440)
                    _min_of_sample = samples.T.min(dim=-1).values.unsqueeze(-1)
                    acq_val = -(samples.T - _min_of_sample <= 1e-10).sum(dim=-1) / _num_sample # Note: will be taken argmin    
                elif acq.lower() == 'qei':
                    _best_y = samples.T.mean(dim=-1).min()
                    acq_val = (samples.T - _best_y).clamp(max=0).mean(dim=-1)    # Note: will be taken argmin      
                
                self.acq_vals.append(acq_val)

        elif acq.lower() in ['pred']:
            # Pure exploitation with predicted mean as the acquisition function.
            for model_idx, (model, likelihood) in enumerate(zip(self.models.models, self.likelihoods.likelihoods)):
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(_test_x))
                    _val = observed_pred.mean
                _val = self._interpolation_calibrate(test_x=_test_x, target_value=_val, cuda=self.cuda, model_idx=model_idx)
                _maxima = self.optimize_action_set(_test_x, _val, model_idx)
                acq_val = _maxima - _val
                self.acq_vals.append(acq_val.unsqueeze(-1))
        elif acq.lower() in ['ucb', 'lcb', 'sur']:
            beta = kwargs.get('beta', 2)
            if beta == 0: # real beta
                beta = real_beta(self.model_num, self.init_x.size(0), delta=.1, horizon=100)
            # Actually LCB in the NE setting: we take the argmin
            for model_idx, (model, likelihood) in enumerate(zip(self.models.models, self.likelihoods.likelihoods)):
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(_test_x))
                    _std2 = observed_pred.stddev.mul_(beta) # default is 2
                    _mean = observed_pred.mean
                    lower, upper = _mean.sub(_std2), _mean.add(_std2)

                lcb = self._interpolation_calibrate(test_x=_test_x, target_value=lower, cuda=self.cuda, model_idx=model_idx)
                ucb = self._interpolation_calibrate(test_x=_test_x, target_value=upper, cuda=self.cuda, model_idx=model_idx)
                if acq.lower() in ['ucb', 'lcb']:
                    _partial_lcb = self.optimize_action_set(_test_x, ucb, model_idx, return_index=False)
                    acq_val = _partial_lcb - ucb # argmin (lcb_partial - ucb global)
                    
                elif acq.lower() in ['sur']:
                    # stepwise uncertainty reduction (https://arxiv.org/abs/1611.02440)
                    # simplified using the known fact that it is equivalent to the uncertainty reduction
                    # raise NotImplementedError(f"acq {acq} not implemented")
                    sigma = ucb - lcb
                    acq_val = -sigma # will be taken the argmin
                self.acq_vals.append(acq_val.unsqueeze(-1))

        elif acq.lower() in ['epsilon_greedy']:
            # adopted from https://arxiv.org/pdf/1804.10586.pdf, changed the UCB part to be pred to make it sound.
            epsilon = kwargs.get('epsilon', 0.1)
            if np.random.rand() < epsilon: # exploitation
                candidate = self.query(_test_x, acq='pred', method=method, no_scaling=True, **kwargs)
            else:
                candidate = self.query(_test_x, acq='sur', method=method, no_scaling=True, **kwargs)
            if not kwargs.get('no_scaling', False):
                candidate = torch.from_numpy(self.scaler.inverse_transform(candidate.unsqueeze(0)).squeeze())
            return candidate

        elif acq.lower() in ['random']:
            self.acq_vals = torch.rand(_test_x.size(0))
        else:
            raise NotImplementedError(f"acq {acq} not implemented")

        self.acq_vals = torch.cat(self.acq_vals, dim=-1).sum(dim=-1) if acq.lower() not in ['random'] else self.acq_vals
        min_pts = torch.argmin(self.acq_vals)
        candidate = _test_x[min_pts]

        if not kwargs.get('no_scaling', False):
            candidate = torch.from_numpy(self.scaler.inverse_transform(candidate.unsqueeze(0)).squeeze())
        return candidate


    def query_cont(self, test_x:torch.Tensor,  acq:str="ts", method:str="love", **kwargs)->torch.Tensor:
        '''
        BO for Nash Equilibrium, optimize the acquisition function in continuous space, then find the closest point in a given discretization.
        Input:
            @test_x: discrete search space
            @acq: acquisition function
            @method: sampling option
        Output:
            candidate: to be evaluated (with minimum gap)
        '''
        # Set into eval mode
        self.models.eval()
        self.likelihoods.eval()

        if self.cuda:
            _test_x = test_x.to(device=DEVICE, type=TYPE)
        else:
            _test_x = test_x.float()
        
        self.acq_vals = []


        if acq.lower() in ['ucb', 'lcb']:
            # Actually LCB in the NE setting: we take the argmin
            def _lcb(_test_x):
                # lower = torch.linalg.norm(_test_x)
                acq_vals = torch.zeros(1)
                model = self.models.models[0]
                likelihood = self.likelihoods.likelihoods[0]
                model.eval()
                likelihood.eval()
                observed_pred = likelihood(model(_test_x))
                lower, upper = observed_pred.confidence_region()
                return lower
        
            # randomly pick one initial point
            x = torch.tensor(test_x[np.random.choice(a=test_x.size(0), size=1)], requires_grad=True, dtype=torch.float)

            # Set hyperparameters for optimization
            convergence_threshold = kwargs.get("convergence_threshold", 1e-6)
            max_iterations = kwargs.get("max_iterations", 1000)
            learning_rate = kwargs.get("learning_rate", 1e-2)

            # Gradient Descent Loop
            for i in range(max_iterations):
                # Compute the gradient of the convex function at the current point.
                with torch.autograd.set_detect_anomaly(True):
                    gradient = torch.autograd.grad(_lcb(x), x)[0]

                # Update the current point.
                x = x - learning_rate * gradient

                # Check for convergence.
                if torch.norm(gradient) < convergence_threshold:
                    break

        else:
            raise NotImplementedError(f"acq {acq} not implemented")
        
        # find the nearest point in the givin discretization
        candidate_idx = find_nearest_points(test_x=test_x, x=x)
        candidate = test_x[candidate_idx]
        return candidate



