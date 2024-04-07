import gpytorch
import torch
import tqdm

from .sgld import SGLD
from typing import Any
from sklearn.preprocessing import RobustScaler, StandardScaler
from ..model import GPRegressionModel, ExactGPRegressionModel, RFFGPRegressionModel, LargeFeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPE = torch.float

class BO():
    """
    BO Pipeline
    """
    def __init__(self, init_x:torch.tensor, 
                init_y:torch.tensor, 
                lr:float=1e-6, 
                train_iter:int=10, 
                spectrum_norm:bool=False,
                verbose:bool=False, 
                robust_scaling:bool=True, 
                low_dim:bool=True,
                gp_type:str='exact_gp',
                noise_constraint:gpytorch.constraints.Interval=None, 
                output_scale_constraints:gpytorch.constraints.Interval=None,
                **kwargs:Any,
                )->None:
        # scale input
        ScalerClass = RobustScaler if robust_scaling else StandardScaler
        self.scaler = ScalerClass().fit(init_x)
        init_x = self.scaler.transform(init_x)
        # init vars
        self.lr = lr
        self.low_dim = low_dim
        self.verbose = verbose
        self.init_x = init_x
        self.init_y = init_y
        self.n_init = init_x.size(0)
        self.data_dim = init_x.size(1)
        assert init_x.size(0) == init_y.size(0)
        self.train_iter = train_iter

        self.spectrum_norm = spectrum_norm
        self.noise_constraint = noise_constraint
        self.scale_constraints = output_scale_constraints
        self.gp_type = gp_type.lower()

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.train_x = self.init_x.to(device=DEVICE, type=TYPE)
            self.train_y = self.init_y.to(device=DEVICE, type=TYPE)
        else:
            self.train_x = self.init_x.clone()
            self.train_y = self.init_y.clone()

        # init model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=self.noise_constraint)
        if  self.gp_type == 'exact_gp':
            self.feature_extractor = LargeFeatureExtractor(self.data_dim, self.low_dim, spectrum_norm=self.spectrum_norm)
            self.model = ExactGPRegressionModel(
                            train_x=self.train_x,
                            train_y=self.train_y, 
                            gp_likelihood=self.likelihoods,
                            gp_feature_extractor=self.feature_extractor, 
                            low_dim=self.low_dim, 
                            output_scale_constraint=self.noise_constraint,
                        )
        elif self.gp_type == 'dk':
            self.model = GPRegressionModel(
                            train_x=self.train_x,
                            train_y=self.train_y, 
                            gp_likelihood=self.likelihoods,
                            low_dim=self.low_dim, 
                            output_scale_constraint=self.noise_constraint,
                        )
        elif self.gp_type == 'rff':
            self.model = RFFGPRegressionModel(
                            train_x=self.train_x,
                            train_y=self.train_y, 
                            gp_likelihood=self.likelihoods,
                            low_dim=self.low_dim, 
                            output_scale_constraint=self.noise_constraint,
                        )
        else:
            raise NotImplementedError(f"{self.gp_type} not implemented")

        self.train()

    def _mll_loss(self, pred:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        assert hasattr(self, 'mll')
        return -self.mll(pred, y)

    def _mse_loss(self, pred:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        tmp_loss = torch.nn.MSELoss()
        return tmp_loss(pred.mean, y)
    
    def train(self, verbose:bool=False, loss_type:str='nll')->None:
        self.model.train()
        self.likelihood.train()

        # optimizer
        if self.gp_type == 'dk':
            params = [
                {'params': self.model.feature_extractor.parameters()},
                {'params': self.model.covar_module.parameters()},
                {'params': self.model.mean_module.parameters()},
                {'params': self.model.likelihood.parameters()},
            ]
            self.optimizer = SGLD(params, lr=self.lr)
        else:
            params = [
                {'params': self.model.covar_module.parameters()},
                {'params': self.model.mean_module.parameters()},
                {'params': self.model.likelihood.parameters()},
            ]
            self.optimizer = torch.optim.Adam(params=params, lr=self.lr)
        
        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        if loss_type.lower() == "nll":
            self.loss_func = self._mll_loss
        elif loss_type.lower() == "mse":
            self.loss_func = self._mse_loss
        else:
            raise NotImplementedError(f"{loss_type} not implemented")
        
        # training iterations
        iterator = tqdm.tqdm(range(self.train_iter)) if verbose else range(self.train_iter)
        for i in iterator:
            # Zero backprop gradients
            self.optimizer.zero_grad()
            # Get output from model
            self.output = self.model(self.train_x)
            # Calc loss and backprop derivatives
            self.loss = self.loss_func(self.output, self.train_y)
            self.loss.backward()
            self.optimizer.step()

            if verbose:
                iterator.set_postfix({"Loss": self.loss.item()})
            
    def query(self, test_x:torch.Tensor,  acq:str="ts", method:str="love", **kwargs)->torch.Tensor:
        '''
        Ordinary BO
        Input:
            @test_x: discrete search space
            @acq: acquisition function
            @method: sampling option
        Output:
            candidate: to be evaluated
        '''
        # Set into eval mode
        self.model.eval()
        self.likelihood.eval()

        if self.cuda:
          _test_x = test_x.cuda()

        if acq.lower() in ["ts", 'qei']:
            # Either Thompson Sampling or Monte-Carlo EI
            _num_sample = 100 if acq.lower() == 'qei' else 1 
            if method.lower() == "love":
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(200):
                    # NEW FLAG FOR SAMPLING
                    with gpytorch.settings.fast_pred_samples():
                        samples = self.model(_test_x).rsample(torch.Size([_num_sample]))
            elif method.lower() == "ciq":
                with torch.no_grad(), gpytorch.settings.ciq_samples(True), gpytorch.settings.num_contour_quadrature(10), gpytorch.settings.minres_tolerance(1e-4):
                        samples = self.likelihood(self.model(_test_x)).rsample(torch.Size([_num_sample]))
            else:
                raise NotImplementedError(f"sampling method {method} not implemented")
            if acq.lower() == 'ts':
                self.acq_val = samples.T.squeeze()
            elif acq.lower() == 'qei':
                _best_y = samples.T.mean(dim=-1).max()
                self.acq_val = (samples.T - _best_y).clamp(min=0).mean(dim=-1)

        elif acq.lower() in ["ucb",'lcb']:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(_test_x))
                lower, upper = observed_pred.confidence_region()
            if acq.lower() == 'ucb':
                self.acq_val = upper
            elif acq.lower() == 'lcb':
                self.acq_val = -lower            

        elif acq.lower() in ['pred']:
            # Pure exploitation with predicted mean as the acquisition function.
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(_test_x))
                self.acq_val = observed_pred.mean
        else:
            raise NotImplementedError(f"acq {acq} not implemented")

        max_pts = torch.argmax(self.acq_val)
        candidate = _test_x[max_pts]
        return candidate


