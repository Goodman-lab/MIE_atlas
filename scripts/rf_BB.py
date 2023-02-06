#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:33:09 2021
Modified May 2022

Uncertainty based on bayesian bootstrapping (bayesboot)

@author: cg588
"""

import sys
try: 
    sys.path.index('/home/cg588')
except ValueError:
    sys.path.append('/home/cg588')

import pandas as pd
import numpy as np

from MIE_atlas.featurizers.molfeaturizer import MorganFPFeaturizer
from MIE_atlas.tests.uncertainty_metrics import *
from MIE_atlas.utils.uncertainty.bayesian_bootstrap import BayesianBootstrapRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


target_list = ['AChE', 'ADORA2A', 'ADRB1', 'ADRB2', 'AR', 'CHRM1', 'CHRM2', 'CHRM3', 'DD1R', 'DD2R', 'EDNRA', 'HRH1', 'HTR2A', 'KCNH2', 'LCK', 'NR3C1', 'OPRD1', 'OPRM1', 'SLC6A2', 'SLC6A3', 'SLC6A4']

bb_test_r2_list = []
bb_test_mae_list = []

bb_val_r2_list = []
bb_val_mae_list = []

bb_test_cali_r2_list = []
bb_val_cali_r2_list = []
bb_test_eff_list = []
bb_val_eff_list = []
bb_test_disp_list = []
bb_val_disp_list = []
bb_test_gmp_list = []
bb_val_gmp_list = []
bb_test_oracle_auc_list = []
bb_val_oracle_auc_list = []

print("Beginning model building...")
print("There are", len(target_list), "targets to build for.")


for target in target_list:

    print("Building uncertainty for target:", target)
    raw_data = pd.read_csv("/home/cg588/MIE_atlas/data/input_data/"+target+".csv")
    val_data = pd.read_csv("/home/cg588/MIE_atlas/data/ext_validation/"+target+".csv")
    
    featurizer = MorganFPFeaturizer(radius = 2, nBits = 8000)
    X = featurizer.transform(raw_data['SMILES'])
    y = raw_data['P(Act)'].values
    X_val = featurizer.transform(val_data['SMILES'])
    y_val = val_data['P(Act)'].values
    
    assert y.shape[0] == X.shape[0]
    assert y_val.shape[0] == X_val.shape[0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    
    
    rf = RandomForestRegressor()
    bb = BayesianBootstrapRegressor(rf, n_replications = 25, resample_size = X_train.shape[0])
    

    bb.fit(X_train, y_train)

    # Get test metrics
    
    y_tpred_bb = bb.predict(X_test)
    
    bb_test_mae = mean_absolute_error(y_test, y_tpred_bb)
    bb_test_r2 = r2_score(y_test, y_tpred_bb)
    
    bb_test_cali_r2, bb_test_eff, bb_test_disp, bb_test_gmp = all_uncertainty_metrics(bb, X_test, y_test)
    
    bb_test_oracle_auc = oracle_error_auc(bb, X_test, y_test, significance=0.05)

    bb_test_mae_list.append(bb_test_mae)
    bb_test_r2_list.append(bb_test_r2)
    
    bb_test_cali_r2_list.append(bb_test_cali_r2)
    bb_test_oracle_auc_list.append(bb_test_oracle_auc)
    bb_test_eff_list.append(bb_test_eff)
    bb_test_disp_list.append(bb_test_disp)
    bb_test_gmp_list.append(bb_test_gmp)
    
    print("test cali curves: bb")
    cali_curve(bb, X_test, y_test)

    # Get val metrics
    

    y_vpred_bb = bb.predict(X_val)


    bb_val_mae = mean_absolute_error(y_val, y_vpred_bb)
    bb_val_r2 = r2_score(y_val, y_vpred_bb)
    
    bb_val_cali_r2, bb_val_eff, bb_val_disp, bb_val_gmp = all_uncertainty_metrics(bb, X_val, y_val)
    
    bb_val_oracle_auc = oracle_error_auc(bb, X_val, y_val, significance=0.2)
    
    bb_val_mae_list.append(bb_val_mae)
    bb_val_r2_list.append(bb_val_r2)
    
    bb_val_cali_r2_list.append(bb_val_cali_r2)
    bb_val_oracle_auc_list.append(bb_val_oracle_auc)
    bb_val_eff_list.append(bb_val_eff)
    bb_val_disp_list.append(bb_val_disp)
    bb_val_gmp_list.append(bb_val_gmp)
    
    print("validation cali curves: bb")
    cali_curve(bb, X_val, y_val)


# Print results to sheet
metrics_list = ["bb test MAE", "bb test R2", "bb test cali R2", "bb test efficiency", "bb test dispersion", "bb test gmean probs", "bb test oracle error AUC", 
                "bb val MAE", "bb val R2", "bb val cali R2", "bb val efficiency", "bb val dispersion", "bb val gmean probs", "bb val oracle error AUC"]

print("END")

results_df = pd.DataFrame(np.column_stack([bb_test_mae_list, bb_test_r2_list, bb_test_cali_r2_list, bb_test_eff_list, bb_test_disp_list, bb_test_gmp_list, bb_test_oracle_auc_list, 
                                           bb_val_mae_list, bb_val_r2_list, bb_val_cali_r2_list, bb_val_eff_list, bb_val_disp_list, bb_val_gmp_list, bb_val_oracle_auc_list]).T, 
                          index = metrics_list, 
                          columns=target_list)

results_mean = results_df.mean(axis = 1)
results_std = results_df.std(axis=1)

results_df['MEAN'] = results_mean
results_df['STD'] = results_std
print(results_df)

from datetime import date
today = date.today().strftime("%d_%m_%Y")
results_df.to_csv("/home/cg588/MIE_atlas/tests/BB_rf/"+today+".csv")

#%%
