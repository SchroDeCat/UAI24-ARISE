'''
Model for RFF
'''
from typing import Any
from .exact_gp import ExactGPRegressionModel
import gpytorch
import torch

class RFFGPRegressionModel(ExactGPRegressionModel):
        def __init__(self, train_x:torch.Tensor, train_y:torch.Tensor, gp_likelihood:gpytorch.likelihoods, 
                     low_dim:bool=True, output_scale_constraint:gpytorch.constraints.Interval=None, **kwargs:Any)->None:
            '''
            Using RFF as the kernel.
            '''
            super(ExactGPRegressionModel, self).__init__(train_x, train_y, gp_likelihood)
            output_scale = output_scale_constraint if output_scale_constraint else gpytorch.constraints.Interval(0.7,5.0)
            num_samples = kwargs.get("num_samples", 10)
            if low_dim:
                self.covar_module = gpytorch.kernels.ScaleKernel( 
                                        gpytorch.kernels.RFFKernel(num_samples=num_samples,
                                            num_dims=train_x.size(-1), 
                                        ), 
                                    outputscale_constraint=output_scale,)
            else:
                self.covar_module = gpytorch.kernels.LinearKernel(num_dims=train_x.size(-1))
            try: # gpytorch 1.6.0 support
                self.mean_module = gpytorch.means.ConstantMean(constant_prior=train_y.mean())
            except Exception: # gpytorch 1.9.1
                self.mean_module = gpytorch.means.ConstantMean()

            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
