'''
Model for exact GP
Shall support multiple acquisition functions:
UCB, TS, (q)EI.
'''
from typing import Any
import gpytorch
import torch

class ExactGPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x:torch.Tensor, train_y:torch.Tensor, gp_likelihood:gpytorch.likelihoods, 
                     low_dim:bool=True, output_scale_constraint:gpytorch.constraints.Interval=None, **kwargs:Any)->None:
            '''
            Exact GP
            '''
            super(ExactGPRegressionModel, self).__init__(train_x, train_y, gp_likelihood)
            output_scale = output_scale_constraint if output_scale_constraint else gpytorch.constraints.Interval(0.7,5.0)
            length_scale = gpytorch.constraints.Interval(0.005, 4.0)
            if low_dim:
                self.covar_module = gpytorch.kernels.ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                                        gpytorch.kernels.MaternKernel(
                                            nu=2.5, 
                                            ard_num_dims=train_x.size(-1), 
                                            lengthscale_constraint=length_scale,
                                        ), 
                                    outputscale_constraint=output_scale,)
            else:
                self.covar_module = gpytorch.kernels.LinearKernel(num_dims=train_x.size(-1))
            try: # gpytorch 1.6.0 support
                self.mean_module = gpytorch.means.ConstantMean(constant_prior=train_y.mean())
            except Exception: # gpytorch 1.9.1
                self.mean_module = gpytorch.means.ConstantMean()

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x:torch.tensor)->gpytorch.distributions.MultivariateNormal:
            self.projected_x = x
            mean_x = self.mean_module(self.projected_x)
            covar_x = self.covar_module(self.projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)