from const import industries, start_date, end_date
from yahooquery import Screener, Ticker
import pandas as pd

class DataLoader:
    def __init__(self, path):
        self.path = path

        # For Loading Tickers
        self.tickers_file_name = 'tickers.txt'
        self.industries = industries
        self.assets_per_sector = 40
        self.tickers_list = []
        self.have_tickers_data = False
        
        with open('{}/{}'.format(path, self.tickers_file_name), 'r') as f:
            for line in f.readlines():
                self.have_tickers_data = True
                self.tickers_list.append(line.strip('\n'))

        # End - Tickers

        # For Loading Price Data
        self.price_file_name = 'prices.csv'
        self.have_price_data = False

        with open('{}/{}'.format(path, self.price_file_name), 'r') as f:
            for line in f.readlines():
                self.have_price_data = True

        if self.have_price_data:
            self.price_data = pd.read_csv('{}/{}'.format(path, self.price_file_name))

        # End - Prices
    
        # For Loading Factor Data
        self.factor_file_name = '3factors.csv'

        if self.have_price_data:
            self.price_data = pd.read_csv('{}/{}'.format(path, self.price_file_name))
            if not 'Mkt-RF' in list(self.price_data.columns):
                self.load_factor_data()
        

        # End - Prices

    '''
    This function will generate a list of tickers to use for analysis.
    '''
    def get_ticker_list(self, saveTrue=True):
        if self.have_tickers_data:
            return self.tickers_list

        screener = Screener()
        for industry in self.industries:
            data = screener.get_screeners(industry, count=self.assets_per_sector)
            for i in range(len(data[industry]['quotes'])):
                self.tickers_list.append(data[industry]['quotes'][i]['symbol'])
        
        if saveTrue:
            with open('{}/{}'.format(self.path, self.tickers_file_name), 'w') as f:
                for ticker in self.tickers_list:
                    f.write(ticker+"\n")

        return self.tickers_list

    def load_asset_data(self):
        if self.have_price_data:
            return self.price_data

        ticker_object = Ticker(self.tickers_list, asynchronous=True)
        self.price_data = ticker_object.history(start=start_date, end=end_date).reset_index()

        # we want to remove symbols that ipo'd after start_date
        counts_by_symbol = self.price_data.groupby('symbol').date.nunique().reset_index()
        DAYS_MUST_HAVE = max(set(list(counts_by_symbol['date'])), key=list(counts_by_symbol['date']).count) # This is the mode of # of days

        symbols_to_drop = list(counts_by_symbol.loc[counts_by_symbol['date'] != DAYS_MUST_HAVE]['symbol'])
        self.price_data = self.price_data[~self.price_data.isin(symbols_to_drop)]
        self.price_data['date'] = pd.to_datetime(self.price_data['date'])

        self.price_data.to_csv("{}/{}".format(self.path, self.price_file_name))

        return self.price_data

    def load_factor_data(self):
        self.factor_data = pd.read_csv("{}/{}".format(self.path, self.factor_file_name))
        self.factor_data['date'] = pd.to_datetime(self.factor_data['Date'], format="%Y%m%d")
        return self.factor_data

