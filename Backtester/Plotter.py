import pandas as pd
import matplotlib.pyplot as plt

def calculateSharpe(portVal):
    portVal['date'] = pd.to_datetime(portVal['date'])
    portVal['RF'] = portVal['RF']/365 # Annual risk free rate
    portVal['ret'] = portVal['Wealth'].pct_change()
    portVal['ExcessRet'] = portVal['ret'] - portVal['RF']
    SR = portVal['ExcessRet'].mean()/portVal['ExcessRet'].std()

    return SR

def PlotWealth(portVal, file):
    portVal.plot(x='date', y='Wealth')
    plt.savefig('{}.png'.format(file))
