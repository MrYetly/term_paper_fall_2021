#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:01:58 2021

@author: ivananich
"""

import pandas as pd
import numpy as np

##############################################################################
#import cutoff data
##############################################################################

pge_cutoffs = pd.read_stata('../data/cutoffs/pgecutoffs.dta')
pge_cutoffs['iou'] = 'pge'
pge_cutoffs = pge_cutoffs.rename(columns={'pge_day': 'day'})

sce_cutoffs = pd.read_stata('../data/cutoffs/scecutoffs.dta')
sce_cutoffs['iou'] = 'sce'
sce_cutoffs = sce_cutoffs.rename(columns = {'sce_day': 'day'})

sdge_cutoffs = pd.read_stata('../data/cutoffs/sdgecutoffs.dta')
sdge_cutoffs['iou'] = 'sdge'
sdge_cutoffs = sdge_cutoffs.rename(columns = {'sdge_day': 'day'})

cutoffs = pd.concat([pge_cutoffs, sce_cutoffs, sdge_cutoffs])

###create date column unifying year, month, day

#first add leading zeros
cutoffs.month = cutoffs.month.apply(lambda x: str(x).zfill(2))
cutoffs.day = cutoffs.day.apply(lambda x: str(x).zfill(2))

#create date column
cutoffs['date'] = cutoffs.apply(
    lambda x: pd.to_datetime(f'{x.year}-{x.month}-{x.day}', format = '%Y-%m-%d'),
    axis = 1,
)

##############################################################################
#import CSI data
##############################################################################

#declare columns to import
cols = [
        'Application Id',
        'Utility',
        'Application Status',
        'App Received Date',
        'App Complete Date',
        'Self Installer',
        'Installer Name',
        'System Size DC',
        'System Size AC',
        'Inverter Quantity 1',
        'Inverter Quantity 2',
        'Inverter Quantity 3',
        'Inverter Quantity 4',
        'Inverter Quantity 5',
        'Generator Quantity 1',
        'Generator Quantity 2',
        'Generator Quantity 3',
        'Generator Quantity 4',
        'Generator Quantity 5',

    ]

#declare how many rows to import (set to None if using all)
rows = None

#import select columns from SDGE
data = pd.read_csv(
    '../data/Interconnected_Project_Sites_2021-09-30/SDGE_Interconnected_Project_Sites_2021-09-30.csv',
    usecols = cols,
    nrows = rows,
)

#append SCE data
data = data.append(
    pd.read_csv(
    '../data/Interconnected_Project_Sites_2021-09-30/SCE_Interconnected_Project_Sites_2021-09-30.csv',
    usecols = cols,
    nrows = rows,
    ), 
    ignore_index = True,
)
#append PCE data
data = data.append(
    pd.read_csv(
    '../data/Interconnected_Project_Sites_2021-09-30/PGE_Interconnected_Project_Sites_2021-09-30.csv',
    usecols = cols,
    nrows = rows,
    ), 
    ignore_index = True,
)

##############################################################################
#Create main regression sample
##############################################################################

#rename columns
data = data.rename(
    columns={
        'Application Id': 'app_id',
        'Utility': 'iou',
        'Application Status': 'app_status',
        'App Received Date': 'date_receive',
        'App Complete Date': 'date_complete',
        'Self Installer': 'self_install',
        'Installer Name': 'installer',
        'System Size DC':'size_dc',
        'System Size AC': 'size_ac',
        'Inverter Quantity 1': 'inv_q_1',
        'Inverter Quantity 2': 'inv_q_2',
        'Inverter Quantity 3': 'inv_q_3',
        'Inverter Quantity 4': 'inv_q_4',
        'Inverter Quantity 5': 'inv_q_5',
        'Generator Quantity 1': 'gen_q_1',
        'Generator Quantity 2': 'gen_q_2',
        'Generator Quantity 3': 'gen_q_3',
        'Generator Quantity 4': 'gen_q_4',
        'Generator Quantity 5': 'gen_q_5',
        }
)

#lower the case of iou
data.iou = data.iou.str.lower()

#convert dates to datetimes
date_cols = [
    'date_receive',
    'date_complete',
]
for col in date_cols:
    data[col] = pd.to_datetime(data[col])
    
#restrict sample to date range of cutoffs
keep = []
for iou in ['pge', 'sdge', 'sce']:
    
    iou_cutoffs = cutoffs.loc[cutoffs.iou == iou]
    earliest = iou_cutoffs.date.min()
    latest = iou_cutoffs.date.max()
    index = data.loc[
            (data.date_receive >= earliest) 
            & (data.date_receive <= latest + pd.Timedelta(90, unit = 'day'))
            & (data.iou == iou)
    ].index
    keep.append(index)
keep_index = pd.Index(keep[0])
keep_index = keep_index.union(keep[1])
keep_index = keep_index.union(keep[2])
data = data.loc[keep_index]

#take out self installers
data = data.loc[data.self_install == 'No']

#add constant for regressions
data['constant'] = 1

#create subsidy column
def find_subsidy(row):
    u = row.iou
    d = row.date_receive
    
    subsidy_df = cutoffs.loc[
        (cutoffs.iou == u)
        & (cutoffs.date <= d)
    ]
    
    if subsidy_df.shape[0] == 0:
        subsidy = 0
    else:
        subsidy = subsidy_df.sort_values('date', ascending = False).iloc[0].subsidyperwatt
    
    return subsidy

data['subsidy'] = data.apply(
    lambda x: find_subsidy(x),
    axis = 1,
)


#create month-year columns for app_receive and app_complete
data['month_year_receive'] = data.date_receive.to_numpy().astype('datetime64[M]')
data['month_year_complete'] = data.date_complete.to_numpy().astype('datetime64[M]')

#Aggregate inverter and generator quantities (still need to handle missing data being treated as 0)
data['inv_q'] = data[['inv_q_1', 'inv_q_2', 'inv_q_3', 'inv_q_4', 'inv_q_5']].fillna(0).sum(axis=1)
data['gen_q'] = data[['gen_q_1', 'gen_q_2', 'gen_q_3', 'gen_q_4', 'gen_q_5']].fillna(0).sum(axis=1)

#replace 0 with NaN where applicable
for col in ['inv', 'gen']:
    sub_df = data[[f'{col}_q_1', f'{col}_q_2', f'{col}_q_3', f'{col}_q_4', f'{col}_q_5']]
    
    #Evaluate whether all entries in row are NaN
    nan_test = np.all(sub_df.isna(), axis = 1)
    all_nan_index = nan_test.loc[nan_test == True].index
    
    #change 0 to NaN according to index
    data.loc[all_nan_index, f'{col}_q'] = np.nan
    

#output sample
sample_cols = [
        'app_id',
        'iou',
        'app_status',
        'date_complete',
        'self_install',
        'size_dc',
        'gen_q',
        'month_year_receive',
        'inv_q',
        'subsidy',
    ]

sample = data[sample_cols]
sample.to_csv('sample.csv')

##############################################################################
#create sample of IOU-month-years
##############################################################################

ioumy = data[['iou', 'month_year_complete']].copy()
ioumy.drop_duplicates(inplace = True)
ioumy.dropna(inplace = True)
ioumy.reset_index(inplace = True, drop = True)


#add constant for regressions
ioumy['constant'] = 1


#find projects completed in IOU during month-year
def find_completed_project_feature(row, feature):
    '''
    returns a feature (column) of a sub dataset of the projects completed in an IOU at any point 
    in a month-year.    
    '''
    m_y = row.month_year_complete
    iou = row.iou
    
    ref_df = data.loc[
            (data.month_year_complete == m_y)
            & (data.iou == iou)
    ]
    
    return ref_df[feature]
    
    
#find concurrent projects in IOU during month-year
def find_concurrent_project_feature(row, feature):
    '''
    returns a feature (column) of a sub dataset of the projects engaged in at any point 
    in a month-year in an IOU. 
    Note: completed projects are counted as concurrent projects
    '''
    m_y = row.month_year_complete
    iou = row.iou
    
    ref_df = data.loc[
            (data.month_year_complete >= m_y)
            & (data.month_year_receive <= m_y)
            & (data.iou == iou)
    ]
    
    return ref_df[feature]
    
outcomes = [
    'size_dc',
    'gen_q',
    'inv_q',
]  

for o in outcomes:
    
    #if any of outcome in iou-month-year obs is not NaN, set obs = (Sum of outcome in iou-month-year) , else set obs = NaN
    #this is done becuase if all of outcome is NaN, sum will be zero instead of NaN otherwise
    ioumy[f'{o}_complete'] = ioumy.apply(
        lambda x: find_completed_project_feature(x, o).sum() if np.any(find_completed_project_feature(x, o).notna()) == True else np.nan,
        axis = 1,
    )
    
    ioumy[f'{o}_concurrent'] = ioumy.apply(
        lambda x: find_concurrent_project_feature(x, o).sum() if np.any(find_concurrent_project_feature(x, o).notna()) == True else np.nan,
        axis = 1,
    )

ioumy = ioumy.rename(
    columns = {
        'month_year_complete': 'month_year',
        'size_dc_complete': 'size_comp',
        'size_dc_concurrent': 'size_conc',
        'gen_q_complete': 'gen_q_comp',
        'gen_q_concurrent': 'gen_q_conc',
        'inv_q_complete': 'inv_q_comp',
        'inv_q_concurrent': 'inv_q_conc',
        },
)

#create columns for total quantity of inverters and generators completed and concurrent
ioumy['q_comp'] = ioumy.apply(
    lambda x: x.gen_q_comp + x.inv_q_comp if np.any(x[['gen_q_comp', 'inv_q_comp']].notna()) else np.nan,
    axis = 1,
)
ioumy['q_conc'] = ioumy.apply(
    lambda x: x.gen_q_conc + x.inv_q_conc if np.any(x[['gen_q_conc', 'inv_q_conc']].notna()) else np.nan,
    axis = 1,
)

ioumy.to_csv('ioumy.csv')



