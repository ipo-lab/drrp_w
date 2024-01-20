from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cvxpy as cp
from util import LoadData, generate_date_list, start, end, factors_list, nearestPD
from Optimizers import Optimizers, GetOptimalAllocation, drrpw_net
from FactorModelling import GetParameterEstimates
import PortfolioClasses as pc
import LossFunctions as lf
from torch.autograd import Variable
import torch

from torch.utils.data import DataLoader

def RunBacktest(path_to_data, opt_type, InitialValue=1000000, lookback = 52, datatype='broad', delta_rob=0.05):
    returns, assets_list_cleaned, prices, factors = LoadData(path_to_data, e2e=True, datatype=datatype)

    nan_columns = prices.columns[prices.isna().any()].tolist()
    prices.drop(nan_columns, axis=1, inplace=True)
    returns.drop(nan_columns, axis=1, inplace=True)

    assets_list_cleaned = [x for x in assets_list_cleaned if not x in nan_columns]
    print("{} Assets".format(len(assets_list_cleaned)))
    holdings = pd.DataFrame(columns=['date']+assets_list_cleaned)
    portVal = pd.DataFrame(columns=['date', 'Wealth'])


    dates = generate_date_list(returns, prices, start=start, end=end)
    first = True
    
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
            n = len(list(currentPrices))
            x = np.ones(n) / n
            first = False
        else:     
            CurrentPortfolioValue = np.dot(currentPrices,noShares)
            portVal.loc[len(portVal)] = [date] + [CurrentPortfolioValue]
            
        # We don't want the current date information, hence the lack of equality
        # Get last 30
        date = str(date)
        lookback = 52
        returns_lastn = returns[(returns['date'] < date)].tail(lookback)
        asset_returns = returns_lastn.drop('date', axis=1)
        factor_returns = factors[(factors['date'] < date)].tail(lookback)
        factor_returns = factor_returns.drop('date', axis=1)

        mu, Q = GetParameterEstimates(asset_returns, factor_returns, log=False, bad=True, shrinkage=(opt_type == Optimizers.RP_Shrinkage))
        Q = nearestPD(Q)

        try:
            x = GetOptimalAllocation(mu, Q, opt_type, x, delta_robust=delta_rob)
        except:
            pass

        # Update Holdings
        holdings.loc[len(holdings)] = [date] + list(x)

        # Update shares held
        # 50% of 100k = 50k. If price is 100 we have 50,000/100=50 shares
        noShares = np.divide(x*CurrentPortfolioValue, currentPrices)
        print('Done {}'.format(date))
    
    portVal['date'] = pd.to_datetime(portVal['date'])
    portVal = portVal.merge(factors[['date','RF']], how='left', on='date')

    return holdings, portVal

