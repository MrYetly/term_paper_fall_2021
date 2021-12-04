#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:45:49 2021

@author: ivananich
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col, Summary
import numpy as np
from matplotlib import pyplot as plt



##############################################################################
#import sample data
##############################################################################

sample = pd.read_csv('sample.csv')

#change column names
sample = sample.rename(
    columns = {
        'date_complete': 'date',
        'month_year_receive': 'myr',
        },
)

#turn dates into datetimes
sample.date = pd.to_datetime(sample.date)

#create time variable based on month
sample.myr = pd.to_datetime(sample.myr)
sample['myr_value'] = sample.myr.map(lambda x: x.value)
sample.myr_value = sample.myr_value/10000000000

#drop NaNs
sample.dropna(inplace = True)

##############################################################################
# create vars
##############################################################################
sample['month'] = sample.date.map(lambda x: str(x.month))
sample['year'] = sample.date.map(lambda x: str(x.year))

months = pd.get_dummies(sample.month, drop_first = True)
years = pd.get_dummies(sample.year, drop_first = True)
ious = pd.get_dummies(sample.iou, drop_first = True)

sample = pd.concat([sample, months, years, ious], axis = 1)

sample['log_size_dc'] = np.log(sample.size_dc)
sample['log_gen_q'] = np.log(sample.gen_q)
sample['log_subsidy'] = np.log(sample.subsidy)

sample['constant']=1

##############################################################################
# Analysis
##############################################################################


iou_res = []
for iou in ['pge', 'sdge', 'sce']:
   
    sub_sample = sample.loc[sample.iou == iou]
    
    endog = sub_sample['log_gen_q']

    #declare regressors
    exog =  list(months.columns) + list(years.columns)
    regressors = [
        'log_subsidy',
        'constant',
        'myr_value',
    ]
    for v in regressors:
        exog.append(v)
    exog = sub_sample[exog]
    
    #OLS
    ols = sm.OLS(endog, exog)
    ols_res = ols.fit()
    iou_res.append(ols_res)
    

sum_table = summary_col(
        iou_res,
        model_names = ['PGE', 'SDGE', 'SCE'],
        regressor_order = ['log_subsidy',],
        drop_omitted = True,
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
    }
    )

print(sum_table)

# with open("log_size_dc_res.tex", 'w') as f:
#         f.write(sum_table.as_latex()[49:-50])

##############################################################################
# Create histogram displaying time trend of # of observations
##############################################################################

bins = list(sample.myr.unique())
bins.sort()



sub_df = sample[['myr', 'subsidy', 'iou']].copy()
sub_df.subsidy = sub_df.subsidy * 1000
colors = {
    'pge': 'red',
    'sce': 'green',
    'sdge': 'purple',
}
for iou in sub_df.iou.unique():
    
    fig, ax = plt.subplots(figsize = (16,8))

    sub_df.loc[sub_df.iou == iou].plot(
        x = 'myr', 
        y = 'subsidy', 
        kind = 'scatter',
        ax = ax,
        color = colors[iou],
    )
    sub_df.loc[sub_df.iou == iou].myr.hist(
        bins = bins, 
        ax = ax, 
        alpha = 0.5,
        color = colors[iou],
    )
    
    ax.set_ylabel('Count/$/Kw')
    ax.set_xlabel('Month')
    ax.set_ylim(0, 3300)
    
    fig.savefig(f"../term paper/app_hist_{iou}.png")







