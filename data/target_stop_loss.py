import pandas as pd
'''
The function returns the following structure of the dataframe (numbers may vary):

    ratio	    ATR    friday_close	  target	  stop_loss
0	NRG_CPRT    0.339	0.334919	  0.673919	  0.221919
1	DE_BXP	    5.762	4.280950	  10.042950	  2.360283
2	NRG_STE	    0.211	0.204631	  0.415631	  0.134298
3	SO_NLOK	    2.499	3.405268	  5.904268	  2.572268
4	PEG_O	    1.092	0.951858	  2.043858	  0.587858

It contains the average true range (ATR) for each ratio in units of ratio
(not %) and the closing price of Friday. The 'target' column equals the
'friday_close' column + the 'ATR' column; the 'stop_loss' columns equals the
'friday_close' column - 1/3 * the 'ATR' column.
'''

def target_stop_loss(atr_dict, df):
    '''
    - The function takes the atr_dict (dictionary) which is also used in the
      empirical-performance.py file. It is converted into a df named 'atr_df'

    - The second argument, df, is the sample of the 10 best ratios over the last
    6 weeks (now only 7). It contains the daily ratios of the last 2 years.
    '''
    atr_df = pd.DataFrame(atr_dict.items(), columns=['ratio', 'ATR'])
    #get the closing price of Friday
    friday_close = pd.Series(dtype='float64')

    for i in range(0, len(atr_df)):
        friday_close = friday_close.append(pd.Series(df[atr_df.ratio[i]].iloc[-1]))

    friday_close = friday_close.rename('friday_close')
    index_list = list(range(0, len(friday_close)))
    friday_close.index = index_list
    atr_df = pd.concat([atr_df, friday_close], axis=1)
    # add target price columns
    atr_df['target'] = atr_df['ATR'] + atr_df['friday_close']
    # add stop/loss price column
    atr_df['stop_loss'] = atr_df['friday_close'] - (1/3 * atr_df['ATR'])

    return atr_df
