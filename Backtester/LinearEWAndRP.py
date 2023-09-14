from enum import Enum
import numpy as np
import cvxpy as cp
import math
from util import nearestPD
from scipy.stats import gmean
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
from Optimizers import RP, EW
import os


class LinearEWAndRPNet(nn.Module):
    def __init__(self, n_x, n_y, n_obs, lr=0.001, cache_path='cache/'):
        super(LinearEWAndRPNet, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1).uniform_(0, 1))

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        # Define performance loss
        self.perf_loss = lf.sharpe_loss

    # Get RP Portfolio
    # Get EW Portfolios
    # Construct New Portfolio = alpha*EW + (1-alpha)*RP
    # Do gradient descent to learn alpha
    def forward(self, X, observations, batching=False):
        # Covariance Matrix
        Q = np.cov(observations.cpu().detach().numpy(), rowvar=False)
        mu = np.ones((self.n_y, 1)) # Neither EW or RP care about mu so just make it 1s for length

        EW_portfolio = torch.tensor(EW(mu, Q), dtype=torch.float)
        RP_portfolio = torch.tensor(RP(mu, Q), dtype=torch.float)

        optimal_portfolio = self.alpha*EW_portfolio + (1-self.alpha)*RP_portfolio
        
        return optimal_portfolio, []

   #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, train_set, date, epochs=None, lr=None, opt=None, batching=False):
        self.optimizer=opt
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

        # Number of elements in training set
        n_train = len(train_set)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        losses = []
        # Train the neural network
        for epoch in range(epochs):
            # TRAINING: forward + backward pass
            train_loss = 0
            self.optimizer.zero_grad()
            
            for t, (x, y, y_perf) in enumerate(train_set):
                # Forward pass: predict and optimize
                z_star, _ = self(x.squeeze(), y.squeeze())
                loss = (1/n_train) * self.perf_loss(z_star.float(), y_perf.squeeze().float())
                
                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()
            
            losses.append(train_loss)

            grad_value = self.alpha.grad
            # Update parameters
            self.optimizer.step()
            
            # Ensure that gamma, delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name=='alpha':
                    param.data.clamp_(min=0.0001, max=0.9999)


class LinearEWAndRPGridSearch():
    def __init__(self, n_x, n_y, n_obs, lr=0.001, cache_path='cache/'):
        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        self.alpha_try = np.linspace(0, 1, 100)

        # Define performance loss
        self.perf_loss = lf.sharpe_loss

    def sharpe_ratio(self, returns, portfolio):
        return np.mean(returns @ portfolio) / np.std(returns @ portfolio)

   #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, observations):
        best_sharpe = -1e99

        Q = np.cov(observations, rowvar=False)
        mu = np.ones((self.n_y, 1)) # Neither EW or RP care about mu so just make it 1s for length

        EW_portfolio = EW(mu, Q)
        RP_portfolio = RP(mu, Q)

        for alpha in self.alpha_try:
            port = alpha*(EW_portfolio) + (1-alpha)*RP_portfolio
            sharpe_ratio = self.sharpe_ratio(observations, port)
            if sharpe_ratio >= best_sharpe:
                best_sharpe = max(sharpe_ratio, best_sharpe)
                self.opt_port = port
                self.opt_alpha = alpha
        print("Best Sharpe: {}. Best Portfolio: {}. Best Alpha: {}. EW Sharpe: {}. RP Sharpe: {}.".format(best_sharpe, self.opt_port, self.opt_alpha, self.sharpe_ratio(observations, EW_portfolio), self.sharpe_ratio(observations, RP_portfolio)))