import pandas as pd
import numpy as np
import torch
import cvxpy as cp
import os
import math
from cvxpylayers.torch import CvxpyLayer
import time
from util import BatchUtils
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def LoadData(path_to_data, e2e=True, datatype='broad'):
    if e2e:
        path_to_returns = r'{}\asset_weekly_{}.pkl'.format(path_to_data, datatype)
        path_to_prices = r'{}\assetprices_weekly_{}.pkl'.format(path_to_data, datatype)
        path_to_factors = r'{}\factor_weekly_{}.pkl'.format(path_to_data, datatype)

        returns = pd.read_pickle(path_to_returns)
        prices = pd.read_pickle(path_to_prices)
        factors = pd.read_pickle(path_to_factors)

        assets_list = prices.columns.to_list()

        returns = returns.reset_index()
        prices = prices.reset_index()
        factors = factors.reset_index()

        factors = factors.rename(columns={"Date": "date", "Mkt-RF": "RF"})
        factors = factors[['date'] + factors_list]

        return returns, assets_list, prices, factors

    path_to_prices = r'{}\prices.csv'.format(path_to_data)
    path_to_factors = r'{}\3factors.csv'.format(path_to_data)

    prices = pd.read_csv(path_to_prices)
    factors = pd.read_csv(path_to_factors)

    assets_list = list(prices['symbol'].unique())

    assets_list_cleaned = [x for x in assets_list if str(x) != 'nan']
    pivot_prices = np.round(pd.pivot_table(prices, values='close', 
                                    index='date', 
                                    columns='symbol', 
                                    aggfunc=np.mean),2)
    pivot_prices = pivot_prices.reset_index()
    pivot_prices['date'] = pd.to_datetime(pivot_prices['date'])
    factors['date'] = pd.to_datetime(factors['Date'], format="%Y%m%d")

    pivot_prices = pivot_prices.set_index('date')
    returns = pivot_prices.pct_change()
    pivot_prices = pivot_prices.reset_index()
    returns = returns.reset_index()
    returns = returns.merge(factors, on='date', how='left')
    returns = returns.drop(['Date'], axis=1)
    returns = returns.dropna()

    return returns, assets_list_cleaned, pivot_prices, []

def drrpw_nominal_learnDelta(Q):
    n_y = Q.shape[-1]
    # Variables
    phi = cp.Variable((n_y,1), nonneg=True)
    t = cp.Variable()
    
    # Size of uncertainty set
    delta = cp.Parameter(nonneg=True)
    # T = cp.Parameter((n_y, n_y), PSD=True)

    # Norm for x. TODO set this to be the Mahalanobis Norm
    p = 2

    # Kappa, dont need this to be trainable as the value of this doesnt really matter
    k = 2

    # We need to compute \sqrt{x^T Q x} intelligently because
    # cvxpy does not compute well with the \sqrt

    # To do this, I will take the Cholesky decomposition
    # Q = LL^T
    # Then, take the 2-norm of L*x

    # Idea: (L_1 * x_1)^2 = Q_1 x_1

    L = np.linalg.cholesky(Q)
    L /= np.linalg.norm(L)

    # Constraints
    constraints = [
        phi >= 0,
        t >= cp.power(cp.norm(L@phi, 2) + delta*cp.norm(phi, p),2)
        # t >= cp.power(cp.norm(L@phi, 2) + cp.norm(T@phi, 2),2)
    ]

    log_term = 0
    for i in range(n_y):
        log_term += cp.log(phi[i])

    # obj = cp.power(cp.norm(L@w, 2) + delta*cp.norm(w, p),2)
    # obj = cp.sum_squares(cp.norm(L@w, 2) + delta*cp.norm(w, p))
    # cp.quad_form(w, Q)
    # obj = cp.quad_form(w, Q) + 2*delta*cp.norm(w,2)*cp.norm(L@w, 2) + cp.norm(w,2)
    # print('using this one')
    # obj = 2*delta*cp.norm(w,2)*cp.norm(L@w_tilde, 2)
    obj = (t) - k*log_term

    # Objective function
    objective = cp.Minimize(obj)    

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints=constraints)

    return CvxpyLayer(problem, parameters=[delta], variables=[phi, t])

def drrpw_nominal_learnDelta_batched(n_y):
    # Variables
    phi = cp.Variable((n_y,1), nonneg=True)
    t = cp.Variable()
    
    # Size of uncertainty set
    delta = cp.Parameter(1, nonneg=True)

    # L - Square Root of Covariance Matrix
    L = cp.Parameter((n_y, n_y))

    # Norm for x
    p = 2

    # Kappa, dont need this to be trainable as the value of this doesnt really matter
    k = 2

    # Constraints
    constraints = [
        phi >= 0,
        t >= cp.power(cp.norm(L@phi, 2) + delta*cp.norm(phi, p),2)
        # t >= cp.power(cp.norm(L@phi, 2) + cp.norm(T@phi, 2),2)
    ]

    log_term = 0
    for i in range(n_y):
        log_term += cp.log(phi[i])

    obj = (t) - k*log_term

    # Objective function
    objective = cp.Minimize(obj)    

    # Construct optimization problem and differentiable layer
    problem = cp.Problem(objective, constraints=constraints)

    return CvxpyLayer(problem, parameters=[delta, L], variables=[phi, t])

class drrpw_custom(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, delta, Q):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        batch_size, num_assets, num_assets = Q.size()
        opt_layer = drrpw_nominal_learnDelta_batched(num_assets)
        z_stars, _ = opt_layer(delta, Q)

        # Make each portfolio sum to 1
        sums = z_stars.sum(dim=1, keepdim=True)
        z_stars = z_stars / sums

        # Save vectors
        ctx.Q = Q
        ctx.z_stars = z_stars
        ctx.delta_val = delta

        return z_stars

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        time_start = time.time()
        print("In Backward Here")
 
        z_stars = ctx.z_stars
        Q_batched = ctx.Q
        delta_batched = ctx.delta_val

        batch_size, num_assets, _ = z_stars.size()

        grads = []
        for batch in range(batch_size):
            z_star = z_stars[batch]
            Q = Q_batched[batch]
            delta = delta_batched[batch]

            grad = 0

            # Common Terms
            risk = z_star.T @ (Q @ z_star)
            phi_norm = np.linalg.norm(z_star)
            m1 = 1/(2*math.sqrt(delta))
            
            for i in range(num_assets):
                grad += m1*(((z_star[i]*z_star[i]) * math.sqrt(risk) / phi_norm) + (phi_norm * ((Q @ z_star)[i] * z_star[i] / risk))) + z_star[i]
            
            grads.append(grad)
        # i=0
        # grad = m1*(((z_star[i]*z_star[i]) * math.sqrt(risk) / phi_norm) + (phi_norm * ((Q @ z_star)[i] * z_star[i] / risk))) + z_star[i]

        grad_input = grad_output.clone()
        return_tuple = tuple([grad_input.sum(dim=1)*g for g in grad]) + (None, None)
        print("In Function Time: {}".format(time.time() - time_start))
        return return_tuple

path_to_data = r"C:\Users\Rafay\Documents\thesis3\thesis\ActualWork\e2e\cache"
factors_list = ['RF']
datatype='cross_asset'
returns, assets_list_cleaned, prices, factors = LoadData(path_to_data, e2e=True, datatype=datatype)
lookback = 104
date = '2019-06-02'
returns_lastn = returns[(returns['date'] < date)].tail(lookback)
asset_returns = returns_lastn.drop(['date'], axis=1)

batch_utils = BatchUtils()
window_size = 52
batched_tensor = batch_utils.convert_to_sw_batched(asset_returns, window_size)
performance_tensor = batch_utils.convert_performance_periods(asset_returns, window_size)
Q = batch_utils.compute_covariance_matrix(batched_tensor)

drrpw = drrpw_custom()
drrpw_layer = drrpw.apply
delta = torch.nn.Parameter(torch.FloatTensor(1).uniform_(0, 0.1))
batch_size = Q.size()[0]
batched_delta = torch.ones((batch_size,1))*delta
batched_delta = batched_delta.type(torch.FloatTensor)
start_time_fwd = time.time()
z = drrpw_layer(batched_delta, Q)
print(z.size())
out = (z - torch.zeros(z.size())).pow(2)
stop_time_fwd = time.time()

start_time_bwd = time.time()
out.sum().backward()
stop_time_bwd = time.time()
print("Backward Time: {}".format(stop_time_bwd - start_time_bwd))
print('Custom Layer Portfolio: {}. Delta Grad : {}. Forward Time: {}. Backward Time: {}'.format(z, delta.grad, stop_time_fwd - start_time_fwd,stop_time_bwd - start_time_bwd))


start_time = time.time()
delta2 = torch.nn.Parameter(torch.FloatTensor(1).uniform_(0, 0.1))

start_time_fwd = time.time()
opt_layer = drrpw_nominal_learnDelta(Q)
z_star, t = opt_layer(delta2)
z_star = torch.divide(z_star, torch.sum(z_star))
out = (z_star - torch.zeros(z_star.size())).pow(2)

stop_time_fwd = time.time()

start_time_bwd = time.time()
out.sum().backward()

print('CvxpyLayer Portfolio: {}. Delta Grad : {}. Forward Time: {}. Backward Time: {}'.format(z_star.detach().numpy()[0], delta2.grad, stop_time_fwd - start_time_fwd, time.time() - start_time_bwd))

