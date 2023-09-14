import pandas as pd
import numpy as np
from datetime import datetime
import torch

start, end = '2016-01-01', '2022-12-31'
# start, end = '2015-01-01', '2015-01-31'
factors_list = ['RF', 'SMB', 'HML']
factors_list = ['RF']

class BatchUtils:
    def __init__(self) -> None:
        pass

    def convert_to_sw_batched(self, input_numpy, window_size):
        # convert numpy array to tensor
        input_tensor = torch.from_numpy(input_numpy.to_numpy())

        # calculate number of training points
        num_training_points = input_tensor.size(0) - window_size + 1

        # initialize tensor to hold sliding windows
        sliding_windows = torch.zeros((num_training_points, window_size, input_tensor.size(1)))

        # populate tensor with sliding windows
        for i in range(num_training_points):
            sliding_windows[i] = input_tensor[i:i+window_size]
        
        return sliding_windows

    def convert_performance_periods(self, input_numpy, window_size):
        input_tensor = torch.from_numpy(input_numpy.to_numpy())
        num_training_points = input_tensor.size(0) - window_size + 1

        # define label tensor shape
        label_shape = (num_training_points, input_tensor.size(1), 1)

        # initialize label tensor to zeros
        label_set = torch.zeros(label_shape)

        # populate label tensor with next time period returns
        for i in range(num_training_points-1):
            label_set[i:i+1] = input_tensor[i+window_size:i+window_size+1].unsqueeze(2)

        return label_set
    
    def compute_covariance_matrix(self, sliding_windows):
        # get the number of batches and number of assets
        num_batches, window_size, num_assets = sliding_windows.size()

        # initialize covariance matrix tensor
        covariance_matrix_tensor = torch.zeros((num_batches, num_assets, num_assets))

        # compute covariance matrix for each batch
        for i in range(num_batches):
            batch = sliding_windows[i]
            Q = torch.Tensor(np.cov(batch.numpy().T))
            try:
                L = np.linalg.cholesky(Q)
            except:
                Q = nearestPD(Q)
                L = np.linalg.cholesky(Q)
            L /= np.linalg.norm(L)
            covariance_matrix_tensor[i] = torch.from_numpy(L)

        return covariance_matrix_tensor

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

def shrinkage(returns):
    """Shrinks sample covariance matrix towards constant correlation unequal variance matrix.
    Ledoit & Wolf ("Honey, I shrunk the sample covariance matrix", Portfolio Management, 30(2004),
    110-119) optimal asymptotic shrinkage between 0 (sample covariance matrix) and 1 (constant
    sample average correlation unequal sample variance matrix).
    Paper:
    http://www.ledoit.net/honey.pdf
    Matlab code:
    https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffde5e2d4e/covCor.m.zip
    Special thanks to Evgeny Pogrebnyak https://github.com/epogrebnyak
    :param returns:
        t, n - returns of t observations of n shares.
    :return:
        Covariance matrix, sample average correlation, shrinkage.
    """
    t, n = returns.shape
    mean_returns = np.mean(returns, axis=0, keepdims=True)
    returns -= mean_returns
    sample_cov = returns.transpose() @ returns / t

    # sample average correlation
    var = np.diag(sample_cov).reshape(-1, 1)
    sqrt_var = var ** 0.5
    unit_cor_var = sqrt_var * sqrt_var.transpose()
    average_cor = ((sample_cov / unit_cor_var).sum() - n) / n / (n - 1)
    prior = average_cor * unit_cor_var
    np.fill_diagonal(prior, var)

    # pi-hat
    y = returns ** 2
    phi_mat = (y.transpose() @ y) / t - sample_cov ** 2
    phi = phi_mat.sum()

    # rho-hat
    theta_mat = ((returns ** 3).transpose() @ returns) / t - var * sample_cov
    np.fill_diagonal(theta_mat, 0)
    rho = (
        np.diag(phi_mat).sum()
        + average_cor * (1 / sqrt_var @ sqrt_var.transpose() * theta_mat).sum()
    )

    # gamma-hat
    gamma = np.linalg.norm(sample_cov - prior, "fro") ** 2

    # shrinkage constant
    kappa = (phi - rho) / gamma
    shrink = max(0, min(1, kappa / t))

    # estimator
    sigma = shrink * prior + (1 - shrink) * sample_cov

    return sigma, average_cor, shrink

def generate_date_list(data, data2, start, end):
    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)

    # Must be in this list
    must = data2.date.apply(lambda x: x.date()).unique().tolist()

    # Train model from start_date to date
    mask = (data['date'] >= start) & (data['date'] <= end) & data['date'].isin(must)

    data = data.loc[mask]
    return data.date.apply(lambda x: x.date()).unique().tolist()

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3