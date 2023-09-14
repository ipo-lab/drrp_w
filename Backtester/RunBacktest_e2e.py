from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cvxpy as cp
from util import LoadData, generate_date_list, start, end, factors_list, BatchUtils
from Optimizers import Optimizers, GetOptimalAllocation, drrpw_net
from FactorModelling import GetParameterEstimates
import PortfolioClasses as pc
import LossFunctions as lf
from torch.autograd import Variable
import torch
from mvo_learn_norm import mvo_norm_net
from LinearEWAndRP import LinearEWAndRPNet
from torch.utils.data import DataLoader

batch_utils = BatchUtils()

def RunBacktest_e2e(path_to_data, opt_type, InitialValue=1000000, lookback = 30, datatype='broad'):
    returns, assets_list_cleaned, prices, factors = LoadData(path_to_data, e2e=True, datatype=datatype)
    holdings = pd.DataFrame(columns=['date']+assets_list_cleaned)
    portVal = pd.DataFrame(columns=['date', 'Wealth'])

    dates = generate_date_list(returns, prices, start=start, end=end)
    first = True

    # Subtract 1 from n_x and n_y since we have a date column
    n_x, n_y, n_obs, perf_period = factors.shape[1] - 1, returns.shape[1] - 1, 40, 11
    lookback = 104
    print("# Factors: {}. # Assets: {}".format(n_x, n_y))

    # Hyperparameters
    lr = 0.01
    epochs_per_date = 25

    if opt_type==Optimizers.CardinalityRP:
        cardinality=10

    # For replicability, set the random seed for the numerical experiments
    set_seed = 200

    if opt_type==Optimizers.MVONormTrained:
        batching = False
        net = mvo_norm_net(n_x, n_y, n_obs, 
                learnT=((opt_type==Optimizers.DRRPWTTrained) or (opt_type==Optimizers.MVONormTrained)), learnDelta=(opt_type==Optimizers.DRRPWDeltaTrained), 
                set_seed=set_seed, opt_layer='nominal').double()
    elif opt_type==Optimizers.LinearEWAndRPOptimizer:
        batching = False
        net = LinearEWAndRPNet(n_x, n_y, n_obs)
    else:
        batching = True
        net = drrpw_net(n_x, n_y, n_obs,
                learnT=(
                        (opt_type==Optimizers.DRRPWTTrained)
                        or (opt_type==Optimizers.MVONormTrained)
                        or (opt_type==Optimizers.DRRPWTTrained_Diagonal)),
                learnDelta=(opt_type in [Optimizers.DRRPWDeltaTrained, Optimizers.DRRPWDeltaTrainedCustom]),
                set_seed=set_seed, T_Diagonal=(opt_type==Optimizers.DRRPWTTrained_Diagonal), use_custom_fwd=(opt_type==Optimizers.DRRPWDeltaTrainedCustom)).double()

    delta_trained = []
    loss_values = []
    grad_values = []
    time_values = []
    fwdtime_values = []
    bwdtime_values = []
    T_diagonals = []
    T_offdiagonals = []

    for date in dates:
        # Get Asset Prices for Today
        currentPrices = (prices[prices['date']==str(date)]
            .drop('date',axis=1)
            .values
            .flatten())
        
        # Update Portfolio Value
        if first:
            portVal.loc[len(portVal)] = [date] + [InitialValue]
            CurrentPortfolioValue = InitialValue
            first = False
        else:     
            CurrentPortfolioValue = np.dot(currentPrices,noShares)
            portVal.loc[len(portVal)] = [date] + [CurrentPortfolioValue]
            
        # We don't want the current date information, hence the lack of equality
        # Get last 30
        date = str(date)
        
        returns_lastn = returns[(returns['date'] < date)].tail(lookback)
        asset_returns = returns_lastn.drop('date', axis=1)

        factor_returns = factors[(factors['date'] < date)].tail(lookback)
        factor_returns = factor_returns.drop('date', axis=1)

        performance_tensor = None
        if batching:
            window_size = 52
            batched_tensor = batch_utils.convert_to_sw_batched(asset_returns, window_size)
            performance_tensor = batch_utils.convert_performance_periods(asset_returns, window_size)
            train_set = (batched_tensor, performance_tensor)
        else:
            train_set = DataLoader(pc.SlidingWindow(factor_returns, asset_returns, n_obs, 
                                                perf_period))

        # net_train to get optimal delta
        net.net_train(train_set, lr=lr, epochs=epochs_per_date, date=date, batching=batching)

        factor_ret_tensor = Variable(torch.tensor(factor_returns.values, dtype=torch.double))
        asset_ret_tensor = Variable(torch.tensor(asset_returns.values, dtype=torch.double))

        # perform forward pass to get optimal portfolio
        x_tensor, _ = net(performance_tensor, asset_ret_tensor, batching=False)
        x = x_tensor.detach().numpy().flatten()
        if opt_type in [Optimizers.DRRPWDeltaTrained, Optimizers.DRRPWDeltaTrainedCustom]:
            delta_val = net.delta.item()
            delta_trained.append(delta_val)
            loss_values.append(net.loss_val)
            time_values.append(net.avg_time)
            fwdtime_values.append(net.avg_fwd_time)
            bwdtime_values.append(net.avg_bwd_time)
            # grad_values.append(net.curr_gradient)
        elif opt_type in [Optimizers.DRRPWTTrained, Optimizers.DRRPWTTrained_Diagonal]:
            T = net.T.detach().numpy()
            delta_trained.append(np.mean(np.diag(T)))
            loss_values.append(np.mean(T - np.diag(np.diag(T))))

        # mu, Q = GetParameterEstimates(asset_returns, factor_returns, log=False, bad=True)
        # x = GetOptimalAllocation(mu, Q, opt_type)

        # Update Holdings
        holdings.loc[len(holdings)] = [date] + list(x)

        # Update shares held
        # 50% of 100k = 50k. If price is 100 we have 50,000/100=50 shares
        print("x: {}. CurrentPortfolioValue: {}. currentPrices: {}".format(x, CurrentPortfolioValue, currentPrices))
        noShares = np.divide(x*CurrentPortfolioValue, currentPrices)
        print('Done {}'.format(date))
                
    portVal['date'] = pd.to_datetime(portVal['date'])
    portVal = portVal.merge(factors[['date','RF']], how='left', on='date')


    return holdings, portVal, [delta_trained, loss_values, grad_values, time_values, fwdtime_values, bwdtime_values]
