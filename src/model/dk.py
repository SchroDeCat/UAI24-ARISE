import gpytorch
import torch

from typing import Any, Union
from botorch.posteriors.gpytorch import GPyTorchPosterior


class GPRegressionModel(gpytorch.models.ExactGP):
        '''
        Feature Extractor + Base Kernel
        '''
        def __init__(self, 
                     train_x:torch.Tensor, 
                     train_y:torch.Tensor, 
                     gp_likelihood:gpytorch.likelihoods,
                     gp_feature_extractor:torch.nn.Sequential, 
                     low_dim:bool=True, 
                     output_scale_constraint:gpytorch.constraints.Interval=None,
                     **kwargs:Any,
                     )->gpytorch.distributions.MultivariateNormal:
            
            super(GPRegressionModel, self).__init__(train_x, train_y, gp_likelihood)
            self.feature_extractor = gp_feature_extractor
            output_scale = output_scale_constraint if output_scale_constraint else gpytorch.constraints.Interval(0.7,5.0)
            try: # gpytorch 1.6.0 support
                self.mean_module = gpytorch.means.ConstantMean(constant_prior=train_y.mean())
            except Exception: # gpytorch 1.9.1
                self.mean_module = gpytorch.means.ConstantMean()
            if low_dim:
                self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1), 
                    outputscale_constraint=output_scale,),
                    num_dims=1, grid_size=100)
            else:
                self.covar_module = gpytorch.kernels.LinearKernel(num_dims=10)

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            self.projected_x = self.feature_extractor(x)
            self.projected_x = self.scale_to_bounds(self.projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(self.projected_x)
            covar_x = self.covar_module(self.projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        def posterior(
            self,
            X: torch.Tensor,
            observation_noise: Union[bool, torch.Tensor] = False,
            **kwargs: Any,
        ) -> GPyTorchPosterior:
            r"""Computes the posterior over model outputs at the provided points.

            Args:
                X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                    of the feature space and `q` is the number of points considered
                    jointly.
                observation_noise: If True, add the observation noise from the
                    likelihood to the posterior. If a Tensor, use it directly as the
                    observation noise (must be of shape `(batch_shape) x q`).

            Returns:
                A `GPyTorchPosterior` object, representing a batch of `b` joint
                distributions over `q` points. Includes observation noise if
                specified.
            """
            self.eval()  # make sure model is in eval mode
            mvn = self.forward(X.float())

            posterior = mvn
            return posterior
        
        def _set_transformed_inputs(self):
            pass
