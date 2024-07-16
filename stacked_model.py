# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from aepsych.models import GPRegressionModel
from typing import Union, Optional, Tuple, Mapping, Any, OrderedDict
import gpytorch
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior
from aepsych.models import GPClassificationModel
import torch
from gpytorch.likelihoods import BernoulliLikelihood
import numpy as np


class StackedRTModel(GPClassificationModel):

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        RTs: torch.Tensor,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        inducing_size: int = 100,
        inducing_point_method: str = "auto",
    ):
        # Augument bounds for added RT feature
        RTs_trans = torch.log(RTs)
        lb = torch.cat((lb, torch.tensor([RTs_trans.min()])))
        ub = torch.cat((ub, torch.tensor([RTs_trans.max()])))
        super().__init__(
            lb=lb,
            ub=ub,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=BernoulliLikelihood(),
            inducing_size=inducing_size,
            inducing_point_method=inducing_point_method,
        )
        # Define the RT submodel
        min_noise = 1e-6
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(
                min_noise, transform=None, initial_value=1e-3
            ),
            noise_prior=GammaPrior(0.9, 10.0),
        )
        self.submodel = GPRegressionModel(
            lb=lb[:-1],
            ub=ub[:-1],
            likelihood=likelihood,
        )

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        warmstart_hyperparams: bool = False,
        warmstart_induc: bool = False,
        **kwargs,
    ) -> None:
        # Append RTs as a feature
        train_rt = torch.log(train_y.abs())
        train_x_aug = torch.cat((train_x, train_rt.unsqueeze(-1)), dim=1)
        # Fit classification model with these data
        super().fit(
            train_x=train_x_aug,
            train_y=(torch.Tensor(train_y)>0).float(),
            warmstart_hyperparams=warmstart_hyperparams,
            warmstart_induc=warmstart_induc,
            **kwargs
        )
        # Fit the RT submodel
        self.submodel.fit(
            train_x=train_x,
            train_y=train_rt,
        )

    def sample(
        self, x: Union[torch.Tensor, np.ndarray], num_samples: int
    ) -> torch.Tensor:
        # Predict RT at x
        rt_f, _ = self.submodel.predict(x)
        # Augment x with predicted RT
        x_samp = torch.cat((x, rt_f.unsqueeze(-1)), dim=-1)
        # Sample
        return super().sample(x=x_samp, num_samples=num_samples)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        if x.shape[-1] < self.lb.shape[-1]:
            # Predict RT at x
            rt_f, _ = self.submodel.predict(x)
            # Augment x with predicted RT
            x_til = torch.cat((x, rt_f.unsqueeze(-1)), dim=-1)
        else:
            x_til = x
        # forward
        return super().forward(x=x_til)
