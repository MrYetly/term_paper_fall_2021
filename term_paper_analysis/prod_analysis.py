#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:45:49 2021

@author: ivananich
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import numpy as np
from matplotlib import pyplot as plt

##############################################################################
#import sample and IOU-month-year data
##############################################################################

ioumy = pd.read_csv('ioumy.csv')

#turn dates into datetimes
ioumy.month_year = pd.to_datetime(ioumy.month_year)

##############################################################################
#estimate productivity of installers in each IOU
##############################################################################

#declare IOUs to analyze
ious = [
        'sdge',
        'sce',
        'pge'
]

#declare regressions in dictionary for later query
regressions = {
    'Size DC (Kw)': {
        'endog': 'size_comp',
        'exog': ['size_conc', 'constant'],
        },
    # 'Generator Quantity': {
    #     'endog': 'gen_q_comp',
    #     'exog': ['gen_q_conc', 'constant'],
    #     },
    # 'Inverter Quantity': {
    #     'endog': 'inv_q_comp',
    #     'exog': ['inv_q_conc', 'constant'],
    #     },
    'Total Quantity': {
        'endog': 'q_comp',
        'exog': ['q_conc', 'constant'],
        },
}
 
#create new names for vars in summary output table
new_names = {
    'size_conc': 'Size Committed To',
    'q_conc': 'Quantity Committed To',
    'constant': 'Constant',
}

for reg_title, var in regressions.items():
    
    #create empty list to OLS results  
    prod_res = []
    prod_res_names = []
    
    #find max values among IOUS to set consistent limits to axes for plots below
    y_max = ioumy[var['endog']].max()
    x_max = ioumy[var['exog']].iloc[:,0].max()
    
    for iou in ious:
        #select IOU sub sample, copy to enable addition of fitted values later
        sub_ioumy = ioumy.loc[ioumy.iou == iou][var['exog'] + [var['endog']]].copy()
        w_nans = sub_ioumy.shape[0]
        #drop NaN
        sub_ioumy.dropna(inplace = True)
        wo_nans = sub_ioumy.shape[0]
        print(f'NaNs for {reg_title} in {iou}: {w_nans - wo_nans} out of {w_nans}')

        
        #get endogenous and exogenous variables
        endog = sub_ioumy[var['endog']]
        exog = sub_ioumy[var['exog']]
        
        #OLS
        ols = sm.OLS(endog, exog)
        ols_res = ols.fit()
        
        #add to lists for summary table
        prod_res.append(ols_res)
        prod_res_names.append(iou)
        
        #rename regressors for summary table
        names = ols_res.model.exog_names
        for i in range(len(names)):
            name = names[i]
            new_name = new_names[name]
            names[i] = new_name
            
        #add fitted values to sub_ioumy dataframe for plots
        sub_ioumy['fitted'] = ols_res.fittedvalues
        

        #create plot with best fit line
        fig, ax = plt.subplots(figsize = (8,6))
        sub_ioumy.plot(
            x = exog.iloc[:,0].name, 
            y = endog.name, 
            kind = 'scatter', 
            ax = ax,
        )
        sub_ioumy.plot(
            x = exog.iloc[:,0].name, 
            y = 'fitted', 
            kind = 'line', 
            ax = ax, 
            color='red', 
            legend = False,
        )
        ax.set_ylabel(f'{reg_title} Installed Per Month')
        ax.set_xlabel(f'Concurrent {reg_title} Committed To Per Month')
        ax.set_title(f'{iou}, {reg_title} (N Months = {sub_ioumy.shape[0]})')
        ax.set_aspect('equal')
                
        #find max values to set limits to axes
        ax.set_xlim(0, 1.1*x_max)
        ax.set_ylim(0,1.1*y_max)
        
        #export figures
        fig.savefig(f"../term paper/{var['endog']}_{iou}.png")

    #create summary table
    sum_table = summary_col(
        prod_res,
        model_names = prod_res_names,
        regressor_order = ols_res.model.exog_names,
        drop_omitted = True,
        info_dict={
        'N':lambda x: "{0:d}".format(int(x.nobs)),
    }
    )
    
    with open(f"{var['endog']}_res.tex", 'w') as f:
        f.write(sum_table.as_latex()[49:-25])


