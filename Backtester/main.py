from RunBacktest import RunBacktest
from RunBacktest_e2e import RunBacktest_e2e
from Optimizers import Optimizers
from Plotter import calculateSharpe,PlotWealth
import pickle
import numpy as np

path_to_data = r"C:\Users\Rafay\Documents\thesis3\thesis\ActualWork\e2e\cache"
path_to_results = r"C:\Users\Rafay\Documents\thesis3\thesis\ActualWork\Results"
InitialValue = 1000000 # $1,000,000
lookback = 52 # Number of days preceeding current date to train

datatype = 'cross_asset_fixed'

opt_try = [Optimizers.DRRPWDeltaTrained, Optimizers.DRRPWDeltaTrained]
# opt_try = [Optimizers.DRRPWDeltaTrained]
opt_try = [Optimizers.DRRPWTTrained_Diagonal, Optimizers.DRRPWTTrained]

for opt_type in opt_try:
    print(opt_type.value)
    if opt_type in [Optimizers.DRRPWDeltaTrained, Optimizers.DRRPWDeltaTrainedCustom, Optimizers.DRRPWTTrained, Optimizers.LearnMVOAndRP, Optimizers.MVONormTrained, Optimizers.DRRPWTTrained_Diagonal, Optimizers.LinearEWAndRPOptimizer]:
        holdings, portVal, hyperparams = RunBacktest_e2e(path_to_data, opt_type, InitialValue, lookback, datatype=datatype)
        if opt_type in [Optimizers.DRRPWDeltaTrained, Optimizers.DRRPWDeltaTrainedCustom]:
            with open(path_to_results + '{}_deltavals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[0], f)
            with open(path_to_results + '{}_lossvals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[1], f)
            with open(path_to_results + '{}_gradvals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[2], f)
            with open(path_to_results + '{}_timevals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[3], f)
            with open(path_to_results + '{}_fwdtimevals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[4], f)
            with open(path_to_results + '{}_bwdtimevals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[5], f)
        if opt_type in [Optimizers.DRRPWTTrained, Optimizers.DRRPWTTrained_Diagonal]:
            with open(path_to_results + '{}_diagonalvals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                pickle.dump(hyperparams[0], f)
            with open(path_to_results + '{}_offdiagonalvals_{}.pkl'.format(opt_type.value, datatype), 'wb') as f:
                print(hyperparams[1])
                pickle.dump(hyperparams[1], f)

        portVal.to_pickle(path_to_results + '{}_{}_value.pkl'.format(opt_type.value, datatype, lookback))
        holdings.to_pickle(path_to_results + '{}_{}_holdings.pkl'.format(opt_type.value, datatype, lookback))
    else:
        if opt_type==Optimizers.DRRPW:
            for d in [100]:
                holdings, portVal = RunBacktest(path_to_data, opt_type, InitialValue, lookback, datatype=datatype, delta_rob=d)
                name = "{}_delta{}".format(opt_type.value, d)
                print(name)

                portVal.to_pickle(path_to_results + '{}_{}_value.pkl'.format(name, datatype, lookback))
                holdings.to_pickle(path_to_results + '{}_{}_holdings.pkl'.format(name, datatype, lookback))
        else:
            holdings, portVal = RunBacktest(path_to_data, opt_type, InitialValue, lookback, datatype=datatype)
            portVal.to_pickle(path_to_results + '{}_{}_value.pkl'.format(opt_type.value, datatype, lookback))
            holdings.to_pickle(path_to_results + '{}_{}_holdings.pkl'.format(opt_type.value, datatype, lookback))
    

    SharpeRatio = calculateSharpe(portVal)

    print('{} SR: '.format(opt_type.value, SharpeRatio))
