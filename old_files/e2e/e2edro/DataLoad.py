# DataLoad module
#
####################################################################################################
## Import libraries
####################################################################################################
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import pandas_datareader as pdr
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time
from scipy.stats import levy_stable, norm

####################################################################################################
# TrainTest class
####################################################################################################
class TrainTest:
    def __init__(self, data, n_obs, split):
        """Object to hold the training, validation and testing datasets

        Inputs
        data: pandas dataframe with time series data
        n_obs: Number of observations per batch
        split: list of ratios that control the partition of data into training, testing and 
        validation sets. 
    
        Output. TrainTest object with fields and functions:
        data: Field. Holds the original pandas dataframe
        train(): Function. Returns a pandas dataframe with the training subset of observations
        """
        self.data = data
        self.n_obs = n_obs
        self.split = split

        n_obs_tot = self.data.shape[0]
        numel = n_obs_tot * np.cumsum(split)
        self.numel = [round(i) for i in numel]

    def split_update(self, split):
        """Update the list outlining the split ratio of training, validation and testing
        """
        self.split = split
        n_obs_tot = self.data.shape[0]
        numel = n_obs_tot * np.cumsum(split)
        self.numel = [round(i) for i in numel]

    def train(self):
        """Return the training subset of observations
        """
        return self.data[:self.numel[0]]

    def test(self):
        """Return the test subset of observations
        """
        return self.data[self.numel[0]-self.n_obs:self.numel[1]]

####################################################################################################
# Generate linear synthetic data
####################################################################################################
def synthetic(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=100):
    """Generates synthetic (normally-distributed) asset and factor data

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    np.random.seed(set_seed)

    # 'True' prediction bias and weights
    a = np.sort(np.random.rand(n_y) / 250) + 0.0001
    b = np.random.randn(n_x, n_y) / 5
    c = np.random.randn(int((n_x+1)/2), n_y)

    # Noise std dev
    s = np.sort(np.random.rand(n_y))/20 + 0.02

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) / 50
    X2 = np.random.randn(n_tot, int((n_x+1)/2)) / 50

    # Synthetic outputs
    Y = a + X @ b + X2 @ c + s * np.random.randn(n_tot, n_y)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

####################################################################################################
# Generate non-linear synthetic data
####################################################################################################
def synthetic_nl(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=100):
    """Generates synthetic (normally-distributed) factor data and mix them following a quadratic 
    model of linear, squared and cross products to produce the asset data. 

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    np.random.seed(set_seed)

    # 'True' prediction bias and weights
    a = np.sort(np.random.rand(n_y) / 200) + 0.0005
    b = np.random.randn(n_x, n_y) / 4
    c = np.random.randn(int((n_x+1)/2), n_y)
    d = np.random.randn(n_x**2, n_y) / n_x

    # Noise std dev
    s = np.sort(np.random.rand(n_y))/20 + 0.02

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) / 50
    X2 = np.random.randn(n_tot, int((n_x+1)/2)) / 50
    X_cross = 100 * (X[:,:,None] * X[:,None,:]).reshape(n_tot, n_x**2)
    X_cross = X_cross - X_cross.mean(axis=0)

    # Synthetic outputs
    Y = a + X @ b + X2 @ c + X_cross @ d + s * np.random.randn(n_tot, n_y)

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

####################################################################################################
# Generate non-linear synthetic data
####################################################################################################
def synthetic_NN(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=45678):
    """Generates synthetic (normally-distributed) factor data and mix them following a 
    randomly-initialized 3-layer neural network. 

    Inputs
    n_x: Integer. Number of features
    n_y: Integer. Number of assets
    n_tot: Integer. Number of observations in the whole dataset
    n_obs: Integer. Number of observations per batch
    split: List of floats. Train-validation-test split as percentages (must sum up to one)
    set_seed: Integer. Used for replicability of the numpy RNG.

    Outputs
    X: TrainValTest object with feature data split into train, validation and test subsets
    Y: TrainValTest object with asset data split into train, validation and test subsets
    """
    np.random.seed(set_seed)

    # Syntehtic features
    X = np.random.randn(n_tot, n_x) * 10 + 0.5
    
    # Initialize NN object
    synth = synthetic3layer(n_x, n_y, set_seed).double()

    # Synthetic outputs
    Y = synth(torch.from_numpy(X))

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y.detach().numpy()) / 10
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

####################################################################################################
# E2E neural network module
####################################################################################################
class synthetic3layer(nn.Module):
    """End-to-end DRO learning neural net module.
    """
    def __init__(self, n_x, n_y, set_seed):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model

        Output
        e2e_net: nn.Module object 
        """
        super(synthetic3layer, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        torch.manual_seed(set_seed)

        # Neural net with 3 hidden layers 
        self.pred_layer = nn.Sequential(nn.Linear(n_x, int(0.5*(n_x+n_y))),
                    nn.ReLU(),
                    nn.Linear(int(0.5*(n_x+n_y)), int(0.6*(n_x+n_y))),
                    nn.ReLU(),
                    nn.Linear(int(0.6*(n_x+n_y)), n_y),
                    nn.ReLU(),
                    nn.Linear(n_y, n_y))

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the synthetic3layer NN
    #-----------------------------------------------------------------------------------------------
    def forward(self, X):
        """Forward pass of the NN module

        Inputs
        X: Features. (n_obs x n_x) torch tensor with feature timeseries data

        Outputs
        Y: Syntheticly generated output. (n_obs x n_y) torch tensor of outputs
        """
        Y = torch.stack([self.pred_layer(x_t) for x_t in X])

        return Y

####################################################################################################
# Synthetic data with Gaussian and exponential noise terms
####################################################################################################
def synthetic_exp(n_x=5, n_y=10, n_tot=1200, n_obs=104, split=[0.6, 0.4], set_seed=123):

    np.random.seed(set_seed)
    
    # Exponential (shock) noise term
    exp_noise = 0.2 * np.random.choice([-1,0,1], p=[0.15, 0.7, 0.15], 
                                        size=(n_tot, n_y)) * np.random.exponential(1,(n_tot, n_y))
    exp_noise = exp_noise.clip(-0.3, 0.3)

    # Gaussian noise term
    gauss_noise = 0.2 * np.random.randn(n_tot, n_y)

    # 'True' prediction bias and weights
    alpha = np.sort(np.random.rand(n_y).clip(0.2,1) / 1000)
    beta = np.random.randn(n_x, n_y).clip(-3,3) / n_x

    # Syntehtic features
    X = np.random.randn(n_tot, n_x).clip(-3,3) / 10

    # Synthetic outputs
    Y = (alpha + X @ beta + exp_noise + gauss_noise).clip(-0.2,0.3) / 15

    # Convert to dataframes
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    # Partition dataset into training and testing sets
    return TrainTest(X, n_obs, split), TrainTest(Y, n_obs, split)

####################################################################################################
# Option 4: Factors from Kenneth French's data library and asset data from AlphaVantage
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 
# https://www.alphavantage.co 
####################################################################################################
def AV(start, end, split, freq='weekly', n_obs=104, n_y=None, use_cache=False, save_results=False, 
        AV_key=None):
    """Load data from Kenneth French's data library
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html 

    Inputs
    start: start date
    end: end date
    split: train-validation-test split as percentages 
    freq: data frequency (daily, weekly, monthly)
    n_obs: number of observations per batch
    use_cache: Boolean. State whether to load cached data or download data
    save_results: Boolean. State whether the data should be cached for future use. 

    Outputs
    X: TrainTest object with feature data split into train, validation and test subsets
    Y: TrainTest object with asset data split into train, validation and test subsets
    """

    if use_cache:
        X = pd.read_pickle('./cache/factor_'+freq+'.pkl')
        Y = pd.read_pickle('./cache/asset_'+freq+'.pkl')
    else:
        ticker_str = """AMZN
                    TSLA
                    HD
                    BABA
                    TM
                    MCD
                    NKE
                    LOW
                    SBUX
                    PDD
                    JD
                    BKNG
                    TJX
                    ABNB
                    MELI
                    GM
                    MAR
                    F
                    STLA
                    ORLY
                    RACE
                    CMG
                    AZO
                    LVS
                    HMC
                    HLT
                    LULU
                    ROST
                    SE
                    YUM
                    MBLY
                    DHI
                    CPRT
                    APTV
                    QSR
                    CPNG
                    LEN
                    EBAY
                    ULTA
                    XOM
                    CVX
                    SHEL
                    TTE
                    COP
                    BP
                    EQNR
                    ENB
                    SLB
                    EOG
                    PBR
                    CNQ
                    OXY
                    EPD
                    MPC
                    PXD
                    VLO
                    E
                    WDS
                    PSX
                    SU
                    HES
                    TRP
                    KMI
                    DVN
                    ET
                    WMB
                    CVE
                    HAL
                    MPLX
                    BKR
                    OKE
                    FANG
                    EC
                    TS
                    PBA
                    CTRA
                    MRO
                    TRGP
                    V
                    JPM
                    BAC
                    MA
                    WFC
                    MS
                    SCHW
                    HSBC
                    RY
                    GS
                    AXP
                    C
                    TD
                    HDB
                    SPGI
                    BX
                    BLK
                    PYPL
                    MUFG
                    CB
                    UNH
                    JNJ
                    LLY
                    NVO
                    MRK
                    ABBV
                    PFE
                    TMO
                    AZN
                    ABT
                    DHR
                    NVS
                    BMY
                    AMGN
                    SNY
                    CVS
                    MDT
                    ELV
                    GILD
                    SYK
                    ISRG
                    CI
                    REGN
                    VRTX
                    ZTS
                    GSK
                    HCA
                    BDX
                    BSX
                    MRNA
                    HUM
                    TAK
                    MCK
                    EW
                    A
                    IQV
                    DXCM
                    IDXX
                    BIIB
                    CNC
                    UPS
                    RTX
                    HON
                    CAT
                    UNP
                    BA
                    DE
                    LMT
                    ADP
                    GE
                    CNI
                    CP
                    ITW
                    NOC
                    CSX
                    ETN
                    MMM
                    GD
                    ABB
                    WM
                    NSC
                    TRI
                    FDX
                    EMR
                    ROP
                    PH
                    CTAS
                    JCI
                    PAYX
                    TT
                    TDG
                    ODFL
                    LHX
                    RSG
                    CARR
                    CMI
                    OTIS
                    WCN
                    GWW
                    AME
                    PLD
                    AMT
                    EQIX
                    CCI
                    PSA
                    SPG
                    O
                    VICI
                    WELL
                    DLR
                    SBAC
                    CSGP
                    ARE
                    CBRE
                    EQR
                    AVB
                    WY
                    UDR
                    BEKE
                    EXR
                    VTR
                    MAA
                    INVH
                    SUI
                    WPC
                    IRM
                    ESS
                    PEAK
                    ELS
                    AMH
                    GLPI
                    KIM
                    BPYPP
                    HST
                    CPT
                    AAPL
                    MSFT
                    NVDA
                    TSM
                    ASML
                    AVGO
                    ORCL
                    CSCO
                    ACN
                    ADBE
                    CRM
                    TXN
                    QCOM
                    SAP
                    AMD
                    IBM
                    INTU
                    INTC
                    SONY
                    AMAT
                    NOW
                    ADI
                    INFY
                    FISV
                    UBER
                    LRCX
                    MU
                    SHOP
                    KLAC
                    SNPS
                    SNOW
                    CDNS
                    VMW
                    PANW
                    SQ
                    NXPI
                    WDAY
                    APH
                    ADSK
                    FTNT
                    NEE
                    DUK
                    SO
                    D
                    SRE
                    AEP
                    NGG
                    EXC
                    XEL
                    PCG
                    ED
                    PEG
                    WEC
                    AWK
                    ES
                    CEG
                    EIX
                    FE
                    ETR
                    AEE
                    DTE
                    PPL
                    FTS
                    ELP
                    CNP
                    BEP
                    CMS
                    AES
                    EBR
                    ATO
                    AGR
                    BIP
                    EVRG
                    LNT
                    WTRG
                    NI
        """
        tick_list = ['AAPL', 'MSFT', 'AMZN', 'C', 'JPM', 'BAC', 'XOM', 'HAL', 'MCD', 'WMT', 'COST', 'CAT', 'LMT', 'JNJ', 'PFE', 'DIS', 'VZ', 'T', 'ED', 'NEM']

        tick_list = ticker_str.split('\n')[:-1]
        tick_list = ['PDBC', 'KMLM', 'SPY', 'YINN']
        print(tick_list)

        if n_y is not None:
            tick_list = tick_list[:n_y]

        if AV_key is None:
            print("""A personal AlphaVantage API key is required to load the asset pricing data. If you do not have a key, you can get one from www.alphavantage.co (free for academic users)""")
            AV_key = input("Enter your AlphaVantage API key: ")

        ts = TimeSeries(key=AV_key, output_format='pandas', indexing_type='date')

        # Download asset data
        Y = []
        for tick in tick_list:
            data, _ = ts.get_daily_adjusted(symbol=tick, outputsize='full')
            data = data['5. adjusted close']
            Y.append(data)
            time.sleep(12.5)
        Y = pd.concat(Y, axis=1)
        Y = Y[::-1]
        Y = Y['1999-1-1':end].pct_change()
        Y = Y[start:end]
        Y.columns = tick_list

        # Download factor data 
        dl_freq = '_daily'
        X = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3'+dl_freq, start=start,
                    end=end)[0]
        rf_df = X['RF']
        X = X.drop(['RF'], axis=1)
        mom_df = pdr.get_data_famafrench('F-F_Momentum_Factor'+dl_freq, start=start, end=end)[0]
        st_df = pdr.get_data_famafrench('F-F_ST_Reversal_Factor'+dl_freq, start=start, end=end)[0]
        lt_df = pdr.get_data_famafrench('F-F_LT_Reversal_Factor'+dl_freq, start=start, end=end)[0]

        # Concatenate factors as a pandas dataframe
        X = pd.concat([X, mom_df, st_df, lt_df], axis=1) / 100

        if freq == 'weekly' or freq == '_weekly':
            # Convert daily returns to weekly returns
            Y = Y.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)
            X = X.resample('W-FRI').agg(lambda x: (x + 1).prod() - 1)

        if save_results:
            X.to_pickle('./cache/factor_'+freq+'.pkl')
            Y.to_pickle('./cache/asset_'+freq+'.pkl')

    # Partition dataset into training and testing sets. Lag the data by one observation
    return TrainTest(X[:-1], n_obs, split), TrainTest(Y[1:], n_obs, split)