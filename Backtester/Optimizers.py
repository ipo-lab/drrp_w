from enum import Enum
import numpy as np
import cvxpy as cp
import math
from util import nearestPD, BatchUtils

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import PortfolioClasses as pc
import LossFunctions as lf

from torch.utils.data import DataLoader

class Optimizers(Enum):
    MVO = "MVO"
    RobMVO = "RobMVO"
    RP = "RP"
    RP_Shrinkage = "RP_Shrinkage"
    DRRPW = "DRRPW"
    EW = "EW"
    DRRPWDeltaTrained = "DRRPWDeltaTrained"
    DRRPWDeltaTrainedCustom = "DRRPWDeltaTrainedCustom"
    DRRPWTTrained = "DRRPWTTrained"
    CardinalityRP = "CardinalityRP"
    LearnMVOAndRP = "LearnMVOAndRP"
    MVONormTrained = "MVONormTrained"
    DRRPWTTrained_Diagonal = "DRRPWTTrained_Diagonal"
    LinearEWAndRPOptimizer = "LinearEWAndRPOptimizer"

def GetOptimalAllocation(mu, Q, technique=Optimizers.MVO, x=[], delta_robust=0.05):
    if technique == Optimizers.MVO:
        return MVO(mu,Q)
    if technique in [Optimizers.RP, Optimizers.RP_Shrinkage]:
        return RP(mu, Q)
    if technique == Optimizers.DRRPW:
        return DRRPW(mu, Q, delta=delta_robust)
    if technique == Optimizers.EW:
        return EW(mu, Q)
    if technique == Optimizers.RobMVO:
        return np.array(RobMVO(mu, Q, x))
    if technique == Optimizers.DRRPWDeltaTrained:
        print("Use Other Backtesting Function")

'''
Mean Variance Optimizer
Inputs: mu: numpy array, key: Symbol. value: return estimate
        Q: nxn Asset Covariance Matrix (n: # of assets)
Outputs: x: optimal allocations
'''

def EW(mu, Q):
    n = len(mu)

    return np.ones(n) / n

def MVO(mu,Q):
    
    # # of Assets
    n = len(mu)

    # Decision Variables
    w = cp.Variable(n)
    
    # Target Return for Constraint
    targetRet = np.mean(mu)
    
    constraints = [
        cp.sum(w) == 1, # Sum to 1
        mu.T @ w >= targetRet, # Target Return Constraint
        w>=0 # Disallow Short Sales
    ]

    # Objective Function
    risk = cp.quad_form(w, Q)

    prob = cp.Problem(cp.Minimize(risk), constraints=constraints)
    prob.solve()
    return w.value

import cvxpy as cp
import numpy as np
import numpy as np
from scipy.stats import chisquare
from scipy.stats import gmean
import cvxopt as opt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas
import pandas as pd
options['show_progress'] = False
options['feastol'] = 1e-9

def RobMVO(mu,Q,x0):
    # Penalty on Turnover (very sensitive)
    c = 0
    # Penalty on variance
    lambd = 0.05
    # Pentalty on returns
    rpen = 1
    # Max weight of an asset
    max_weight = 1
    # between 0% and 200%
    turnover = 2
    #size of uncertainty set
    ep = 2

    T = np.shape(mu)[0]
    Theta = np.diag(np.diag(Q))/T
    sqrtTh = np.diag(matrix(np.sqrt(Theta)))
    n = len(Q)

    # Make Q work for abs value
    Q = matrix(np.block([[Q, np.zeros((n,n)), np.zeros((n,n))], [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))], [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))]]))

    # A and B
    b1 = np.ones([1,1])
    try:
        b2 = x0
        b = np.concatenate((b1,b2))
    except:
        b2 = matrix(x0)
        b = np.concatenate((b1,b2))


    A = matrix(np.block([[np.ones(n), c * np.ones(n), -c * np.ones(n)], [np.eye(n), np.eye(n), -np.eye(n)]]))
    

    # G and h
    G = matrix(0.0, (6 * n + 1, 3 * n))
    h = opt.matrix(0.0, (6 * n + 1, 1))
    for k in range(3 * n):
        # xi > 0 constraint
        G[k, k] = -1
    # xi > max_weight
        G[k + 3 * n, k] = 1
        h[k + 3 * n] = max_weight
    for k in range(2 * n):
        # sum dwi+ + dwi- < turnover
        G[6 * n, k + n] = 1

    h[6 * n] = turnover

    quad = lambd*Q

    r = matrix(np.block([rpen*np.array(mu) - ep*sqrtTh, -c * np.ones(2*n)]))

    return np.transpose(np.array(qp(matrix(quad), -1*matrix(r), matrix(G), matrix(h), matrix(A), matrix(b))['x'])[0:n])[0].tolist()
'''
Risk Parity Optimizer
Inputs: mu: numpy array, key: Symbol. value: return estimate
        Q: nxn Asset Covariance Matrix (n: # of assets)
Outputs: x: optimal allocations with equal risk contribution
'''

def RP(mu,Q):
    
    # # of Assets
    n = len(mu)

    # Decision Variables
    w = cp.Variable(n)

    # Kappa
    k = 2
          
    constraints = [
        w>=0 # Disallow Short Sales
    ]

    L = np.linalg.cholesky(Q)
    L /= np.linalg.norm(L)

    # Objective Function
    risk = cp.norm(L@w,2)
    log_term = 0
    for i in range(n):
        log_term += cp.log(w[i])
    
    prob = cp.Problem(cp.Minimize(risk-(k*log_term)), constraints=constraints)
    
    # ECOS fails sometimes, if it does then do SCS
    try:
        prob.solve(verbose=False)
    except:
        prob.solve(solver='SCS',verbose=False)

    x = w.value
    x = np.divide(x, np.sum(x))

    # Check Risk Parity Condition is actually met
    risk_contrib = np.multiply(x, Q.dot(x))
    if not np.all(np.isclose(risk_contrib, risk_contrib[0])):
        print("RP did not work")

    return x

'''
Distributionally Robust Risk Parity With Wasserstein Distance Optimizer
Inputs: mu: numpy array, key: Symbol. value: return estimate
        Q: nxn Asset Covariance Matrix (n: # of assets)
        delta: size of the uncertainty set
Outputs: x: optimal allocations

Formula:
    \min_{\boldsymbol{\phi} \in \mathcal{X}} {(\sqrt{\boldsymbol{\phi}^T \Sigma_{\mathcal{P}}(R)\boldsymbol{\phi}} + \sqrt{\delta}||\boldsymbol{\phi}||_p)^2} - c\sum_{i=1}^n ln(y)

'''

def DRRPW(mu,Q, delta=0):
    
    # # of Assets
    n = len(mu)

    # Decision Variables
    w = cp.Variable(n)

    # Kappa
    k = 100

    # Norm for x
    p = 2

    constraints = [
        w>=0 # Disallow Short Sales
    ]

    # risk = cp.quad_form(w, Q)

    log_term = 0
    for i in range(n):
        log_term += cp.log(w[i])
    
    # We need to compute \sqrt{x^T Q x} intelligently because
    # cvxpy does not compute well with the \sqrt

    # To do this, I will take the Cholesky decomposition
    # Q = LL^T
    # Then, take the 2-norm of L*x

    # Idea: (L_1 * x_1)^2 = Q_1 x_1
    Q = nearestPD(Q)
    L = np.linalg.cholesky(Q)
    L /= np.linalg.norm(L)
    
    obj = cp.power(cp.norm(L@w,2) + math.sqrt(delta)*cp.norm(w, p),2)
    obj = obj - k*log_term

    prob = cp.Problem(cp.Minimize(obj), constraints=constraints)
    
    # ECOS fails sometimes, if it does then do SCS
    try:
        prob.solve(verbose=False)
    except:
        prob.solve(solver='SCS',verbose=False)
    
    x = w.value
    x = np.divide(x, np.sum(x))
    
    # Check Risk Parity Condition is actually met
    # Note: DRRPW will not meet RP, will meet a robust version of RP
    risk_contrib = np.multiply(x, Q.dot(x))

    return x


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

    try:
        L = np.linalg.cholesky(Q)
    except:
        Q = nearestPD(Q)
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

def drrpw_nominal_learnT(n_y, n_obs, Q, isDiagonal=False):
    # Variables
    phi = cp.Variable((n_y,1), nonneg=True)
    t = cp.Variable()

    # Size of uncertainty set
    delta = cp.Parameter(nonneg=True)
    if isDiagonal:
        T_diag = cp.Parameter((n_y, 1), nonneg=True)
        T = cp.diag(T_diag)
        params = T_diag
    else:
        T = cp.Parameter((n_y, n_y), PSD=True)
        params = T

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

    try:
        L = np.linalg.cholesky(Q)
    except:
        Q = nearestPD(Q)
        L = np.linalg.cholesky(Q)

    # Constraints
    constraints = [
        phi >= 0.000000001,
        # t >= cp.power(cp.norm(L@phi, 2) + delta*cp.norm(phi, p),2)
        t >= cp.power(cp.norm(L@phi, 2) + cp.norm(T@phi, 2),2)
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

    return CvxpyLayer(problem, parameters=[params], variables=[phi, t])

def drrpw_nominal_learnT_batched(n_y, isDiagonal=False):
    # Variables
    phi = cp.Variable((n_y,1), nonneg=True)
    t = cp.Variable()

    # Square root of diagonal matrix
    L = cp.Parameter((n_y, n_y))

    if isDiagonal:
        T_diag = cp.Parameter((n_y, 1), nonneg=True)
        T = cp.diag(T_diag)
        params = T_diag
    else:
        T = cp.Parameter((n_y, n_y), PSD=True)
        params = T

    # Norm for x. TODO set this to be the Mahalanobis Norm
    p = 2

    # Kappa, dont need this to be trainable as the value of this doesnt really matter
    k = 2

    # Constraints
    constraints = [
        phi >= 0.000000001,
        # t >= cp.power(cp.norm(L@phi, 2) + delta*cp.norm(phi, p),2)
        t >= cp.power(cp.norm(L@phi, 2) + cp.norm(T@phi, 2),2)
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

    return CvxpyLayer(problem, parameters=[params, L], variables=[phi, t])


from torch.utils.tensorboard import SummaryWriter
import time
class drrpw_net(nn.Module):
    """End-to-end Dist. Robust RP with Wasserstein Distance learning neural net module.
    """
    def __init__(self, n_x, n_y, n_obs, learnT=False, learnDelta=True, set_seed=None, T_Diagonal=False, use_custom_fwd=False):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        net_train: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        opt_layer: String. Determines which CvxpyLayer-object to call for the optimization layer
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'
        train_pred: Boolean. Choose if the prediction layer is learnable (or keep it fixed)
        train_gamma: Boolean. Choose if the risk appetite parameter gamma is learnable
        train_delta: Boolean. Choose if the robustness parameter delta is learnable
        set_seed: (Optional) Int. Set the random seed for replicability

        Output
        drrpw_net: nn.Module object 
        """
        super(drrpw_net, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        self.trainT = learnT
        self.isTDiagonal = T_Diagonal
        self.trainDelta = learnDelta

        self.batch_utils = BatchUtils()

        # Upper/Lower Bound for Delta
        self.ub = 0.01
        self.lb = 0

        # Define performance loss
        self.perf_loss = lf.sharpe_loss

        # Record the model design: nominal, base or DRO
        # Register 'delta' (ambiguity sizing parameter) for DR layer
        if self.trainDelta:
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(self.lb, self.ub))
            self.delta.requires_grad = True

        if self.trainT:                
            Sigma_k = torch.rand(self.n_y, self.n_y)
            Sigma_k = torch.mm(Sigma_k, Sigma_k.t())
            Sigma_k.add_(torch.eye(self.n_y))

            if self.isTDiagonal:
                Sigma_k = torch.rand(n_y, 1)
                # Sigma_k = torch.diag(Sigma_k)

            self.T = nn.Parameter(Sigma_k)
            self.T.requires_grad = True
        
        self.custom_fwd = use_custom_fwd
    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y, batching=False):
        """
        Forward pass of the NN module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        solver_args = {'solve_method': 'SCS'}

        # Covariance Matrix
        if batching:
            Q = self.batch_utils.compute_covariance_matrix(Y)
        else:
            Q = np.cov(Y.cpu().detach().numpy(), rowvar=False)

        param = None
        if self.trainT:
            param = self.T
            d = 0
            
            if batching:
                print('-- Batching --')
                batch_size, num_assets, num_assets = Q.size()
                opt_layer = drrpw_nominal_learnT_batched(self.n_y, isDiagonal = self.isTDiagonal)
                z_stars, _ = opt_layer(param.repeat(batch_size, 1, 1).type(torch.FloatTensor), Q)

                # Make each portfolio sum to 1
                sums = z_stars.sum(dim=1, keepdim=True)
                z_stars = z_stars / sums

                return z_stars, []
            else:
                opt_layer = drrpw_nominal_learnT(self.n_y, self.n_obs, Q, isDiagonal = self.isTDiagonal)

        if self.trainDelta:
            param = self.delta
            d = 1
            if batching:
                batch_size, num_assets, num_assets = Q.size()
                batched_delta = torch.ones((batch_size,1))*self.delta
                batched_delta = batched_delta.type(torch.FloatTensor)

                if self.custom_fwd:
                    drrpw = drrpw_custom()
                    drrpw_layer = drrpw.apply
                    opt_layer = drrpw_layer
                    z_stars, _ = opt_layer(batched_delta, Q, self.z_star.repeat(batch_size, 1, 1))
                else:
                    opt_layer = drrpw_nominal_learnDelta_batched(num_assets)
                    z_stars, _ = opt_layer(batched_delta, Q)
                
                # Make each portfolio sum to 1
                sums = z_stars.sum(dim=1, keepdim=True)
                z_stars = z_stars / sums

                return z_stars, []
            else:
                opt_layer = drrpw_nominal_learnDelta(Q)
        
        z_star, _ = opt_layer(param, solver_args=solver_args)
        z_star = torch.divide(z_star, torch.sum(z_star))

        return z_star, []

    #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, train_set, val_set=None, epochs=None, lr=None, date=None, batching=False):
        """Neural net training module
        
        Inputs
        train_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        val_set: SlidingWindow object containing feaatures x, realizations y and performance
        realizations y_perf
        epochs: Number of training epochs
        lr: learning rate

        Output
        Trained model
        (Optional) val_loss: Validation loss
        """

        # Assign number of epochs and learning rate
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        writer = SummaryWriter(log_dir='board_data/cvxpylayers/{}'.format(date))

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        self.loss_val = 0

        training, y_perf = train_set

        # mu (3x1) -> mu.T (1x3)
        y_perf = torch.transpose(y_perf, 1, 2)

        avg_time = None
        avg_fwd_time = None
        avg_bwd_time = None
        self.avg_time = None
        
        self.z_star = torch.ones((training.size()[-1], 1)) / training.size()[-1]

        for epoch in range(epochs):
            start_time = time.time()
            optimizer.zero_grad()

            # Get optimal allocation
            forward_start = time.time()
            self.z_star, _ = self(None, training, batching=True)
            forward_end = time.time()

            # Compute loss
            loss = self.perf_loss(self.z_star, y_perf)
            
            backward_start = time.time()

            # backward
            loss.backward(retain_graph=True)

            backward_end = time.time()

            self.loss_val = loss.item()

            # Update parameters
            optimizer.step()

            # Ensure that delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name=='T':
                    # param.data.clamp_(0.000001, 1)
                    param.data /= torch.linalg.norm(param.data)
                elif name=='delta':
                    param.data.clamp_(0.000001)

            log = False
            if log:
                writer.add_scalar("Loss/train", loss.item(), epoch)
                writer.add_scalar("Grad/delta", self.delta.grad, epoch)
                writer.add_scalar("Value/delta", self.delta.item(), epoch)
                writer.flush()
            
            if avg_time is not None:
                avg_time = (avg_time + (time.time() - start_time))/2
                avg_fwd_time = (avg_fwd_time + (forward_end - forward_start))/2
                avg_bwd_time = (avg_bwd_time + (backward_end - backward_start))/2
            else:
                avg_time = time.time() - start_time
                avg_fwd_time = time.time() - start_time
                avg_bwd_time = backward_end - backward_start
            

        self.avg_time = avg_time
        self.avg_fwd_time = avg_fwd_time
        self.avg_bwd_time = avg_bwd_time
        print("Average Time per Epoch: {}. Average Forward Time: {}. Average Backward Time: {}".format(avg_time, avg_fwd_time, avg_bwd_time))

class drrpw_newton_util:
    def __init__(self):
        self.tol = 1e-3
        self.max_it = 100

    def f(self,phi, Sigma, num_assets, delta, k):

        return ((phi.T @ Sigma @ phi)**0.5 + delta*np.linalg.norm(phi))**2 - k*np.sum(torch.log(phi))
    
    def gradient(self,phi, Sigma, num_assets, delta, k):        
        risk = phi.T @ Sigma @ phi
        norm = np.linalg.norm(phi)
        grad_phi = 2*(Sigma@phi + delta*risk/norm*phi + delta*norm/risk*(Sigma @ phi) + (delta**2)*phi) - k/phi

        return grad_phi
    
    def hessian(self, phi, Sigma, num_assets, delta, k):
        norm = np.linalg.norm(phi)
        risk = phi.T @ Sigma @ phi
        sigma_phi = Sigma @ phi

        B = 1/(norm*((risk)**0.5))*(sigma_phi @ phi.T) - norm/((risk)**1.5)*(sigma_phi @ (sigma_phi).T) + norm/((risk)**0.5)*Sigma
        A_mat = (phi @ (sigma_phi).T)/(norm*((risk)**0.5)) + ((risk)**0.5)/norm*torch.eye(num_assets,num_assets) - ((risk)**0.5)/(norm**3) * (phi @ phi.T)

        H = 2*Sigma + 2*(delta*A_mat) + 2*(delta * B) + k*torch.diag(1 / (phi.squeeze()**2)) + 2*(delta ** 2)*torch.eye(num_assets,num_assets)
        return H

class drrpw_custom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta_batched, Q, prev_phi):
        batch_size, num_assets, num_assets = Q.size()
        hessian_invs = torch.zeros((batch_size, num_assets, num_assets))
        z_stars = torch.zeros((batch_size, num_assets, 1))

        util = drrpw_newton_util()
        for batch in range(batch_size):
            
            L = nearestPD(Q[batch])
            Sigma = (L@L.T).float()

            delta = delta_batched[batch]
            phi = prev_phi[batch]
            g = torch.ones((num_assets+1, 1))
            k = 1 + delta

            alpha = 0.5
            beta = 0.95

            # Newton's method to get grad(phi) = 0.
            for it in range(util.max_it):        
                if torch.linalg.norm(g) <= util.tol:
                    break
            
                if torch.any(torch.isnan(phi)):
                    break
       
                g = util.gradient(phi, Sigma, num_assets, delta, k)
                H = util.hessian(phi, Sigma, num_assets, delta, k)

                h_inv = torch.linalg.inv(H)

                phi -= (h_inv@g)
            
            if not torch.all(phi >= 0):
                print('{} for batch {} with delta {} and Sigma {}'.format(phi, batch, delta, Sigma))

            hessian_invs[batch] = h_inv
            z_stars[batch] = phi # Remove lambda

        # opt_layer = drrpw_nominal_learnDelta_batched(num_assets)
        # z_stars, _ = opt_layer(delta_batched, Q)

        # Save vectors
        ctx.Q = Q
        ctx.z_stars = z_stars
        ctx.delta_val = delta_batched
        ctx.hessian_invs = hessian_invs

        return z_stars, None

    @staticmethod
    def backward(ctx, grad_output, other):
        z_stars = ctx.z_stars

        # Make each portfolio sum to 1
        sums = z_stars.sum(dim=1, keepdim=True)
        z_stars = z_stars / sums

        Q_batched = ctx.Q
        delta_batched = ctx.delta_val

        hessian_invs = ctx.hessian_invs

        batch_size, num_assets, _ = z_stars.size()
        grads = torch.zeros((batch_size, num_assets, 1))
        
        for batch in range(batch_size):
            b = np.zeros((num_assets,1))

            Sigma = Q_batched[batch]
            phi = z_stars[batch]
            delta = delta_batched[batch]
            h_inv = hessian_invs[batch]

            norm = np.linalg.norm(phi)
            risk = phi.T @ Sigma @ phi
            sigma_phi = Sigma @ phi

            b = - 2*norm * (sigma_phi) / math.sqrt(risk) - 2*(math.sqrt(risk) / norm)*phi - 2*delta*phi

            grads[batch] = torch.tensor(h_inv @ b.detach().cpu().numpy())

        grad_input = grad_output.clone()
        grad_delta = (torch.transpose(grad_input, 1, 2) @ grads).squeeze(1)
        grad_Q = None
        grad_prev_phi = None

        return grad_delta, grad_Q, grad_prev_phi

class CardinalityLoss(nn.Module):
    def __init__(self, cardinality):
        super(CardinalityLoss, self).__init__()

        self.cardinality = cardinality

    def forward(self, output):
        penalty = 100
        a = np.array([0]*self.cardinality)
        a[-1] = -100
        b = np.array([penalty]*(len(output)-self.cardinality))
        penalty_vec = np.concatenate((b,a))
        penalty_tensor = torch.from_numpy(penalty_vec.astype('float64'))
        loss = torch.dot(output, penalty_tensor)
        return loss

