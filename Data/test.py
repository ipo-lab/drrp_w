import pandas as pd

df = pd.read_csv('results/prices.csv')
print(df['symbol'].unique())
print(len(df['symbol'].unique()))
counts = df.groupby('symbol').date.nunique().reset_index()
print(max(set(list(counts['date'])), key=list(counts['date']).count))


symbols_to_drop = list(counts.loc[counts['date'] != 3775]['symbol'])
print(symbols_to_drop)
# df = df[~df.isin(symbols_to_drop)]
# print(df['symbol'].unique())
# print(len(df['symbol'].unique()))
