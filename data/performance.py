'''
There is a dictioncary named "atr_dict":

{'NRG_CPRT': 0.355,
 'DE_BXP': 5.756,
 'NRG_STE': 0.224,
 'SO_NLOK': 2.42,
 'PEG_O': 1.043}

It contains the average true range (ATR) for each ratio in units of ratio
(not %)
'''

# get the 5 best ratios and assign it to a df that contains the
# past 20 and 60 trading days
# best_ratios = list(atr_dict.keys())
# lst_mth_df = df[best_ratios][-20:]
# lst_three_mth_df = df[best_ratios][-60:]


def create_pls(value, df):
    # creating a profit/loss (pl) dict for one and three weeks
    pl_one_dict = {}
    pl_three_dict = {}
    # the portfolio value was set to 10,000 in this case (could be a selector)
    ptf_value = value
    df = df.drop(columns='Date')
    # calcualte change for 20 trading days
    for col in df.columns:
        delta_one = df[col][19] / df[col][0]
        pl_one_dict[col] = round(ptf_value * delta_one, 2)

    # calcualte change for 60 trading days
    for col in df.columns:
        delta_three = df[col][59] / df[col][0]
        pl_three_dict[col] = round(ptf_value * delta_three, 2)

    return pl_one_dict, pl_three_dict
