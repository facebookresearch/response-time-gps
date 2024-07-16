# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# cd projects/rt-modeling
import logging
logger = logging.getLogger('ax.modelbridge.completion_criterion')
logger.setLevel(logging.ERROR)
logger = logging.getLogger('ax.service.utils.with_db_settings_base')
logger.setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from botorch.exceptions.warnings import OptimizationWarning
warnings.filterwarnings("ignore", category=OptimizationWarning)

import torch
import gpytorch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from gpytorch.likelihoods import  BernoulliLikelihood
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from scipy.stats import norm
from functools import partial
from torch.optim import Adam
from aepsych.kernels import PairwiseKernel
from aepsych.likelihoods.ddm import DDMLikelihood, LapseRateRTLikelihood
from aepsych.distributions import (
    LogNormalDDMDistribution,
    ShiftedGammaDDMDistribution,
    ShiftedLogNormalDDMDistribution,
    ShiftedInverseGammaDDMDistribution,
)
from aepsych.models import GPClassificationModel
from stacked_model import StackedRTModel
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.optim.utils.model_utils import sample_all_priors


##### Construct models

def get_covar(dim, pairwise):
    __default_invgamma_concentration = 4.6
    __default_invgamma_rate = 1.0
    ls_prior = gpytorch.priors.GammaPrior(
        concentration=__default_invgamma_concentration,
        rate=__default_invgamma_rate,
        transform=lambda x: 1 / x,
    )
    ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)
    ls_constraint = gpytorch.constraints.Positive(
        transform=None, initial_value=ls_prior_mode
    )
    if pairwise:
        d = dim // 2
    else:
        d = dim
    covar = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(
            lengthscale_prior=ls_prior,
            lengthscale_constraint=ls_constraint,
            ard_num_dims=d,
        ),
        outputscale_prior=gpytorch.priors.GammaPrior(8, 1), # regularize via outputscale
    )
    if pairwise:
        covar = PairwiseKernel(covar, active_dims=list(range(dim)))
    return covar


def get_stacked_covar(dim, pairwise):
    if pairwise:
        rt_kernel = gpytorch.kernels.RBFKernel(active_dims=[dim])
        pwkernel = get_covar(dim=dim, pairwise=pairwise)
        stacked_covar = gpytorch.kernels.ScaleKernel(rt_kernel * pwkernel)
    else:
        stacked_covar = get_covar(dim=dim+1, pairwise=pairwise)
    return stacked_covar


def get_ddm_model(distribution, RTs, lb, ub, pairwise):
    maxshift = 0.15
    rt_likelihood = LapseRateRTLikelihood(
        base_likelihood=DDMLikelihood(
            distribution=distribution,
            restrict_skew=True,
            max_shift=maxshift,
        ),
        max_rt=RTs.abs().max() + 1
    )
    model = GPClassificationModel(
        lb=lb,
        ub=ub,
        dim=len(lb),
        likelihood=rt_likelihood,
        covar_module=get_covar(len(lb), pairwise),
    )
    model.__name__ = distribution.__name__.split('DDM')[0]
    return model


def get_choice_model(lb, ub, pairwise):
    model = GPClassificationModel(
        lb=lb,
        ub=ub,
        dim=len(lb),
        likelihood=BernoulliLikelihood(),
        covar_module=get_covar(len(lb), pairwise),
    )
    model.__name__ = 'choice only'
    return model


def get_stacked_model(RTs, lb, ub, pairwise):
    model = StackedRTModel(
        lb=lb,
        ub=ub,
        RTs=RTs,
        covar_module=get_stacked_covar(len(lb), pairwise),
    )
    model.__name__ = 'stacked'
    return model


##### Fit RT models

def get_torch_optimizer():
    def cb(parameters, optres):
        if not optres.step % 100:
            print(optres.step, optres.fval)

    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR,
        milestones=[200, 500],
        gamma=0.3,
    )
    options = {
        "step_limit": 5000,
        "optimizer":partial(Adam, lr=0.01),
        "scheduler": lr_scheduler,
        #"callback": cb,
    }
    return fit_gpytorch_mll_torch, options


def fit_with_restarts(model_getter, X, y, n_restarts, optimizer_getter):
    n = y.shape[0]
    mlls = []
    models = []
    for _ in range(n_restarts):
        model = model_getter()
        sample_all_priors(model)
        kwargs = {"warmstart_hyperparams": True}
        if optimizer_getter is not None:
            optimizer, optimizer_kwargs = optimizer_getter()
            kwargs.update({"optimizer": optimizer, "optimizer_kwargs": optimizer_kwargs})
        try:
            model.fit(X, y, **kwargs)
            mll = gpytorch.mlls.VariationalELBO(model.likelihood, model, n)
            model_output = mll.model(*mll.model.train_inputs)
            log_likelihood = mll(model_output, mll.model.train_targets).item()
        except Exception:
            log_likelihood = -np.Inf
        mlls.append(log_likelihood)
        models.append(model)

    return models[np.argmax(mlls)]

##### Data loaders

def load_gait_data():
    # Opening JSON file
    datapath = 'data/gait'
    with open(f'{datapath}/sim3_p01.json', 'r') as fin:
        res = json.load(fin)

    with open(f'{datapath}/used_params.json', 'r') as fin:
        used_params = json.load(fin)

    bounds = torch.tensor([[0.1, 0.001, 0], [0.4, 3, 0.1]], dtype=torch.double)

    # Create tensors from the recorded data
    X1 = torch.tensor([
        used_params[str(i)] for (i, j) in res['X']
    ], dtype=torch.double)
    X2 = torch.tensor([
        used_params[str(j)] for (i, j) in res['X']
    ], dtype=torch.double)
    X = torch.cat((X1, X2), dim=1)

    Y = 1 - (torch.tensor(res['Y'], dtype=torch.long) - 1)  # 1 if X1 >> X2

    RTs = torch.tensor(res['RT'], dtype=torch.double)
    RTs[Y == 0] = -RTs[Y == 0] #sign the reaction time
    pairwise = True
    return X, RTs, bounds, pairwise


def load_csf_data():
    csv_path = './data/psychophysics_data_with_rts.csv'
    df = pd.read_csv(csv_path)
    data_cols = list(range(2, 10))
    # remove the angular columns
    data_cols.pop(2)
    data_cols.pop(-2)
    X = df.iloc[:,data_cols].values
    choices = df.response
    rts = df.reaction_time
    RTs = np.where(choices==0, -rts, rts)
    bounds = torch.tensor(
        [[-1.5, 0], [-1.5, 0], [0, 20], [0.5, 7], [1, 10], [0, 10]],
        dtype=torch.double,
    ).T
    pairwise = False
    return torch.tensor(X), torch.tensor(RTs), bounds, pairwise


##### Run experiment fold

def run_experiment(fold, data_loader, ndata=None, run_id='', torch_optim=True):
    X, RTs, bounds, pairwise = data_loader()
    if pairwise:
        lb, ub = bounds[0].repeat(2), bounds[1].repeat(2)
    else:
        lb, ub = bounds[0], bounds[1]
    if torch_optim:
        optimizer_getter = get_torch_optimizer
    else:
        optimizer_getter = None

    if ndata is None:
        ndata = [
            25,
            50,
            #75,
            100,
            #125,
            150,
            #175,
            200,
            300,
        ]

    results = []
    for i, train_size in enumerate(tqdm(ndata)):
        X_train, X_test, y_train, y_test = train_test_split(
            X, RTs, train_size=train_size, random_state=((fold + 1) * 1000)
        )

        distributions = [
            ShiftedGammaDDMDistribution,
            ShiftedInverseGammaDDMDistribution,
            ShiftedLogNormalDDMDistribution,
            LogNormalDDMDistribution,
        ]

        model_getters = [
            partial(
                get_ddm_model,
                distribution=distribution,
                RTs=y_train.abs(),
                lb=lb,
                ub=ub,
                pairwise=pairwise,
            ) for distribution in distributions
        ]
        model_getters.append(
            partial(get_choice_model, lb=lb, ub=ub, pairwise=pairwise)
        )
        model_getters.append(
            partial(get_stacked_model, RTs=y_train.abs(), lb=lb, ub=ub, pairwise=pairwise)
        )

        for model_getter in model_getters:
            model = model_getter()
            #print(f"running {model.__name__}")
            if model.__name__ == 'choice only':
                y = (y_train > 0).float()
            else:
                y = y_train
            model = fit_with_restarts(
                model_getter=model_getter,
                X=X_train,
                y=y,
                optimizer_getter=optimizer_getter,
                n_restarts=5,
            )
            fsamps = model.sample(torch.Tensor(X_test), num_samples=5000)
            if isinstance(model.likelihood, BernoulliLikelihood):
                pm_choice = norm.cdf(fsamps).mean(0)
                brier = brier_score_loss(y_test>0, pm_choice)
                samp_briers = np.array([brier_score_loss(y_test>0, norm.cdf(f.numpy())) for f in fsamps])
            else:  # A DDM likelihood
                lik = model.likelihood(fsamps)
                pm_rt = lik.base_dist.choice_dist.probs.mean(0)
                brier = brier_score_loss(y_test>0, pm_rt.detach().numpy())
                samp_briers = np.array([brier_score_loss(y_test>0, p.numpy()) for p in lik.base_dist.choice_dist.probs.detach()])

            # evaluate RT predictive quality
            # stacked GP model
            if isinstance(model.likelihood, BernoulliLikelihood):
                if hasattr(model, "submodel"):
                    rt_logp_marginal = model.submodel.posterior(X_test).distribution.log_prob(torch.log(y_test.abs())).item()
                    rt_logp_conditional = None
                else: 
                    # (otherwise it's just vanilla bernoulli, pass)
                    rt_logp_marginal = rt_logp_conditional = None
            else: # DDM model
                # we want p(rt | x) = p(rt | x, choice=0) + p(rt | x, choice=1)
                # p(rt | x, choice=0)
                dist = model.likelihood(fsamps).base_dist
                yes_log_probs = torch.log(dist.choice_dist.probs) + dist.rt_yes_dist.log_prob(torch.abs(y_test))
                no_log_probs = torch.log(1 - dist.choice_dist.probs) + dist.rt_no_dist.log_prob(torch.abs(y_test))
                marginal_logps = torch.logsumexp(torch.stack([yes_log_probs, no_log_probs]), dim=0)
                rt_logp_marginal = marginal_logps.sum().item()
                rt_logp_conditional = dist.log_prob(y_test).sum().item()
                                        


            expected_brier = samp_briers.mean()
            result = {
                'model': model.__name__,
                'train_size': train_size,
                'fold': fold,
                'brier': brier,
                'expected_brier': expected_brier,
                'run_id': run_id,
                "rt_logp_marginal":rt_logp_marginal,
                "rt_logp_conditional":rt_logp_conditional,
            }
            results.append(result)
    return results
