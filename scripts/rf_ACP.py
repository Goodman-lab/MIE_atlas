#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:33:09 2021
Modified May 2022

Uncertainty based on conformal prediction (nonconformist)

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
from nonconformist.acp import AggregatedCp
from nonconformist.icp import IcpRegressor
from nonconformist.nc import NcFactory
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


target_list = ['AChE', 'ADORA2A', 'ADRB1', 'ADRB2', 'AR', 'CHRM1', 'CHRM2', 'CHRM3', 'DD1R', 'DD2R', 'EDNRA', 'HRH1', 'HTR2A', 'KCNH2', 'LCK', 'NR3C1', 'OPRD1', 'OPRM1', 'SLC6A2', 'SLC6A3', 'SLC6A4']


acp_test_r2_list = []
acp_test_mae_list = []

acp_val_r2_list = []
acp_val_mae_list = []

acp_test_cali_r2_list = []
acp_val_cali_r2_list = []
acp_test_eff_list = []
acp_val_eff_list = []
acp_test_disp_list = []
acp_val_disp_list = []
acp_test_gmp_list = []
acp_val_gmp_list = []
acp_test_oracle_auc_list = []
acp_val_oracle_auc_list = []
acp_test_data_sparsity_list = []
acp_val_data_sparsity_list = []

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
    norm_reg_nc = NcFactory.create_nc(rf, normalizer_model=KNeighborsRegressor(n_neighbors = 10, weights = 'distance', metric = 'jaccard'))
    acp = AggregatedCp(IcpRegressor(norm_reg_nc), n_models = 25)
    
    acp.fit(X_train, y_train)

    # Get test metrics
    
    y_tpred_acp = [np.mean(interval) for interval in acp.predict(X_test, significance=1)]
    
    acp_test_mae = mean_absolute_error(y_test, y_tpred_acp)
    acp_test_r2 = r2_score(y_test, y_tpred_acp)
    
    acp_test_cali_r2, acp_test_eff, acp_test_disp, acp_test_gmp = all_uncertainty_metrics(acp, X_test, y_test)
    
    acp_test_oracle_auc = oracle_gmp_auc(acp, X_test, y_test)
    acp_test_data_sparsity = data_sparsity_pearsonr(acp, X_train, y_train, X_test, n_neighbors = X_train.shape[0]//20, distance = 'jaccard')
    
    acp_test_mae_list.append(acp_test_mae)
    acp_test_r2_list.append(acp_test_r2)
    
    acp_test_cali_r2_list.append(acp_test_cali_r2)
    acp_test_oracle_auc_list.append(acp_test_oracle_auc)
    acp_test_data_sparsity_list.append(acp_test_data_sparsity)
    acp_test_eff_list.append(acp_test_eff)
    acp_test_disp_list.append(acp_test_disp)
    acp_test_gmp_list.append(acp_test_gmp)
    
    
    print("test cali curves: acp")
    cali_curve(acp, X_test, y_test)

    # Get val metrics
    

    y_vpred_acp = [np.mean(interval) for interval in acp.predict(X_val, significance=1)]


    acp_val_mae = mean_absolute_error(y_val, y_vpred_acp)
    acp_val_r2 = r2_score(y_val, y_vpred_acp)
    
    acp_val_cali_r2, acp_val_eff, acp_val_disp, acp_val_gmp = all_uncertainty_metrics(acp, X_val, y_val)
    
    acp_val_oracle_auc = oracle_gmp_auc(acp, X_val, y_val)
    acp_val_data_sparsity = data_sparsity_pearsonr(acp, X_train, y_train, X_val, n_neighbors = X_train.shape[0]//20, distance = 'jaccard')
    
    acp_val_mae_list.append(acp_val_mae)
    acp_val_r2_list.append(acp_val_r2)
    
    acp_val_cali_r2_list.append(acp_val_cali_r2)
    acp_val_oracle_auc_list.append(acp_val_oracle_auc)
    acp_val_data_sparsity_list.append(acp_val_data_sparsity)
    acp_val_eff_list.append(acp_val_eff)
    acp_val_disp_list.append(acp_val_disp)
    acp_val_gmp_list.append(acp_val_gmp)
    
    
    print("validation cali curves: acp")
    cali_curve(acp, X_val, y_val)
    print("val data sparsity plot")
    data_sparsity_plot(acp, X_train, y_train, X_val, n_neighbors = X_train.shape[0]//20, distance = 'jaccard')



# Print results to sheet
metrics_list = ["ACP test MAE", "ACP test R2", "ACP test cali R2", "ACP test efficiency", "ACP test dispersion", "ACP test gmean probs", "ACP test oracle error AUC", "ACP test data sparsity pearsonr",
                "ACP val MAE", "ACP val R2", "ACP val cali R2", "ACP val efficiency", "ACP val dispersion", "ACP val gmean probs", "ACP val oracle error AUC", "ACP val data sparsity pearsonr"]

print("END")

results_df = pd.DataFrame(np.column_stack([acp_test_mae_list, acp_test_r2_list, acp_test_cali_r2_list, acp_test_eff_list, acp_test_disp_list, acp_test_gmp_list, acp_test_oracle_auc_list, acp_test_data_sparsity_list, 
                                           acp_val_mae_list, acp_val_r2_list, acp_val_cali_r2_list, acp_val_eff_list, acp_val_disp_list, acp_val_gmp_list, acp_val_oracle_auc_list, acp_val_data_sparsity_list]).T, 
                          index = metrics_list, 
                          columns=target_list)

results_mean = results_df.mean(axis = 1)
results_std = results_df.std(axis=1)

results_df['MEAN'] = results_mean
results_df['STD'] = results_std
print(results_df)

from datetime import date
today = date.today().strftime("%d_%m_%Y")
results_df.to_csv("/home/cg588/MIE_atlas/tests/ACP_rf/"+today+".csv")
