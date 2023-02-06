#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:31 2021
Updated Tue May 3 14:13 2022

Metrics for uncertainty quality

@author: cg588
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import auc
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import TruncatedSVD, PCA
import scipy.stats as st
import matplotlib.pyplot as plt
import tensorflow as tf    


# main metrics

def calibration(uncertainty_model, X_test, y_test): 
    
    # Calibration: For given significance, what % of test compounds are within CI
    
    in_interval_list = []
    alpha_list = []
    for significance in range(99): 
        alpha = (significance+1)/100
        alpha_list.append(1 - alpha)
        if uncertainty_model.__class__.__name__ == "BayesianBootstrapRegressor": 
            intervals = uncertainty_model.predict_central_interval(X_test, alpha=alpha)
            in_interval = np.sum((intervals[:,0]<y_test).astype(int)*(intervals[:,1]>y_test).astype(int))/y_test.shape


        if uncertainty_model.__class__.__name__ == "AggregatedCp": 
            intervals = uncertainty_model.predict(X_test, significance=alpha)
            in_interval = np.sum((intervals[:,0]<y_test).astype(int)*(intervals[:,1]>y_test).astype(int))/y_test.shape
        
        if uncertainty_model.__class__.__name__ == "BayesianNN": 
            intervals = uncertainty_model.predict_central_interval(X_test, n_samples = 500, alpha=alpha)
            in_interval = np.sum((intervals[:,0]<y_test).astype(int)*(intervals[:,1]>y_test).astype(int))/y_test.shape

        
        in_interval_list.append(in_interval)
    return alpha_list, in_interval_list

def cali_curve(uncertainty_model, X_test, y_test): 
    
    alpha_list, in_interval_list = calibration(uncertainty_model, X_test, y_test)
    plt.plot(alpha_list, in_interval_list, lw = 2)
    plt.xlabel("Significance level")
    plt.ylabel("% of labels within significance level")
    plt.plot([0, 1], [0, 1], c = 'k', lw = 2)
    plt.show()


def cali_r2_score(uncertainty_model, X_test, y_test): 
    
    alpha_list, in_interval_list = calibration(uncertainty_model, X_test, y_test)
    score = r2_score(alpha_list, in_interval_list)
    
    return score
    
def all_uncertainty_metrics(uncertainty_model, X_test, y_test): 
    
    cali_r2 = cali_r2_score(uncertainty_model, X_test, y_test)
    
    # only BNN supported for other metrics at the moment! 
    # Goal of this function is to sample once and get all metrics so don't call the other functions
    
    if uncertainty_model.__class__.__name__ == "BayesianBootstrapRegressor": 
        y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test)    
        
    if uncertainty_model.__class__.__name__ == "AggregatedCp": 
        intervals = uncertainty_model.predict(X_test, significance = 0.3173)
        y_mu = [np.mean(interval) for interval in intervals]
        y_sigma = [(interval[1]-interval[0])/2 for interval in intervals]
        

    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test, n_samples = 1000)    
        
    efficiency = np.mean(y_sigma)
    dispersion = np.std(y_sigma)
    
    probs = st.norm.pdf(y_test, loc = y_mu, scale = y_sigma)
    gmean_prob = st.mstats.gmean(probs)
    
    return cali_r2, efficiency, dispersion, gmean_prob
        
    
    
    
#%%

# relationship between uncertainty and mean error
# oracle error: eliminate compounds from the dataset, get MAE of remaining compounds

def oracle_error(uncertainty_model, X_test, y_test, significance, bnn_uncertainty = 'all'): 
    
    # Oracle error: filter out highest error compounds, plot remaining MAE
    # Uncertainty: filter out highest uncertainty compounds, plot remaining MAE
    # for total uncertainty (doesn't work well)
    
    if uncertainty_model.__class__.__name__ == "BayesianBootstrapRegressor": 
        intervals = uncertainty_model.predict_central_interval(X_test, alpha = significance)
    
    if uncertainty_model.__class__.__name__ == "AggregatedCp": 
        intervals = uncertainty_model.predict(X_test, significance = significance)
        
    if uncertainty_model.__class__.__name__ == "BayesianNN":
        if bnn_uncertainty == 'all': 
            intervals = uncertainty_model.predict_central_interval(X_test, n_samples = 1000, alpha = significance)
        if bnn_uncertainty == 'epis': 
            epis, epis_mu, epis_sigma = uncertainty_model.get_epistemic(X_test, n_samples = 1000)      
            y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test, n_samples = 1000)

    curve_data = pd.DataFrame(intervals, columns = ['lower', 'upper'])
    y_pred = curve_data.mean(axis = 1)
    curve_data['y_abs_err'] = np.abs(y_pred - y_test)
    curve_data['CI'] = intervals[:,1] - intervals[:,0]
    norm_mean_error = np.mean(curve_data['y_abs_err'])
    n_compounds = y_pred.shape[0]
    sort_error = curve_data.sort_values(by='y_abs_err')['y_abs_err']
    sort_CI = curve_data.sort_values(by='CI')['y_abs_err']
    
    percentile_list = []
    oracle_list = []
    CI_list = []
    
    for percentile in range(100): 
        percentile_list.append(percentile)
        n_to_mean = int(np.around(n_compounds * (100-percentile)/100))
        oracle_error = np.mean(sort_error[0:n_to_mean])
        CI = np.mean(sort_CI[0:n_to_mean])
        oracle_list.append(oracle_error)
        CI_list.append(CI)
    
    oracle_list = np.nan_to_num(oracle_list, nan=norm_mean_error)
    CI_list = np.nan_to_num(CI_list, nan=norm_mean_error)
    
    return percentile_list, oracle_list, CI_list, norm_mean_error

def oracle_error_curve(uncertainty_model, X_test, y_test, significance): 

    percentile_list, oracle_list, CI_list, norm_mean_error = oracle_error(uncertainty_model, X_test, y_test, significance)
    plt.plot(percentile_list, oracle_list, c = 'b', lw=2)
    plt.plot(percentile_list, CI_list, c = 'c', lw=2)
    plt.xlabel("Percentile of compounds eliminated")
    plt.ylabel("Mean Absolute Error")    
    plt.plot([0,100],[norm_mean_error,norm_mean_error], c = 'k', lw=1)
    plt.show()
    
def oracle_error_auc(uncertainty_model, X_test, y_test, significance): 

    
    percentile_list, oracle_list, CI_list, norm_mean_error = oracle_error(uncertainty_model, X_test, y_test, significance)
    
    oracle_auc = auc(percentile_list, oracle_list)
    norm_auc = norm_mean_error*100 - oracle_auc
    CI_auc = auc(percentile_list, CI_list)
    area_between_curves = (CI_auc - oracle_auc) / norm_auc
    
    return area_between_curves


#%%

# relationship between uncertainty and gmp
# similar to oracle error but uses gmp as a measure of 'error' in probabilistic model

def oracle_gmp(uncertainty_model, X_test, y_test): 
    
    # similar to oracle_error function but with prob instead (doesn't work well)
    # view "error" of a probabilistic model not as y_pred - y_true, but as y_pred_dist.prob(y_true)
    # Oracle GMP: always eliminate compound with lowest probability
    # Uncertainty: always eliminate compound with largest std
    
    if uncertainty_model.__class__.__name__ == "BayesianBootstrapRegressor": 
        y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test)    
    
    if uncertainty_model.__class__.__name__ == "AggregatedCp": 
        intervals = uncertainty_model.predict(X_test, significance = 0.3173)
        y_mu = [np.mean(interval) for interval in intervals]
        y_sigma = [(interval[1]-interval[0])/2 for interval in intervals]
        

    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test, n_samples = 1000)       

    probs = st.norm.pdf(y_test, loc = y_mu, scale = y_sigma)
    curve_data = pd.DataFrame(
        {'probs': probs,
         'y_sigma': y_sigma
            })
    norm_gmp = st.mstats.gmean(probs)
    n_compounds = probs.shape[0]
    sort_probs = curve_data.sort_values(by='probs', ascending=False)['probs']
    sort_std = curve_data.sort_values(by='y_sigma', ascending=True)['probs']

    percentile_list = []
    oracle_list = []
    std_list = []
    
    for percentile in range(100): 
        percentile_list.append(percentile)
        n_to_mean = int(np.around(n_compounds * (100-percentile)/100))
        oracle_gmp = st.mstats.gmean(sort_probs[0:n_to_mean])
        std = st.mstats.gmean(sort_std[0:n_to_mean])
        oracle_list.append(oracle_gmp)
        std_list.append(std)
    
    oracle_list = np.nan_to_num(oracle_list, nan=norm_gmp)
    std_list = np.nan_to_num(std_list, nan=norm_gmp)
    
    return percentile_list, oracle_list, std_list, norm_gmp


def oracle_gmp_curve(uncertainty_model, X_test, y_test): 

    percentile_list, oracle_list, std_list, norm_gmp = oracle_gmp(uncertainty_model, X_test, y_test)
    plt.plot(percentile_list, oracle_list, c = 'b', lw=2)
    plt.plot(percentile_list, std_list, c = 'c', lw=2)
    plt.xlabel("Percentile of compounds eliminated")
    plt.ylabel("GMP of remaining compounds")    
    plt.plot([0,100],[norm_gmp,norm_gmp], c = 'k', lw=1)
    plt.show()
    
def oracle_gmp_auc(uncertainty_model, X_test, y_test): 

    
    percentile_list, oracle_list, std_list, norm_gmp = oracle_gmp(uncertainty_model, X_test, y_test)
    
    oracle_auc = auc(percentile_list, oracle_list)
    norm_auc = norm_gmp*100 - oracle_auc
    std_auc = auc(percentile_list, std_list)
    area_between_curves = (std_auc - oracle_auc) / norm_auc
    
    return area_between_curves


#%%

# old functions for individual metrics
# to get all metrics use all_uncertainty_metrics

def efficiency_score(uncertainty_model, X_test, y_test, significance):
    
    # Efficiency represents the mean size of the uncertainty interval. Smaller is better. 
    # This metric is not normalized. This metric is for the entire model. 

    if uncertainty_model.__class__.__name__ == "BayesianBootstrapRegressor": 
        y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test)       
    
    if uncertainty_model.__class__.__name__ == "AggregatedCp": 
        intervals = uncertainty_model.predict(X_test, significance = significance)  
    
    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        intervals = uncertainty_model.predict_central_interval(X_test, n_samples=1000, alpha = significance)
    
    interval_size = intervals[:,1] - intervals[:,0]
    eff_score = np.mean(interval_size)
    
    return eff_score
    
def dispersion_score(uncertainty_model, X_test, y_test, significance): 
    
    # Dispersion represents the standard deviation of the uncertainty intervals. 
    # Larger implies better differentiation between different predictions. 
    # This metric is for the entire model. 
    
    if uncertainty_model.__class__.__name__ == "BayesianBootstrapRegressor": 
        y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test)   

    if uncertainty_model.__class__.__name__ == "AggregatedCp": 
        intervals = uncertainty_model.predict(X_test, significance = significance)  
    
    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        intervals = uncertainty_model.predict_central_interval(X_test, n_samples=1000, alpha = significance)

    interval_size = intervals[:,1] - intervals[:,0]
    disp_score = np.std(interval_size)
    
    return disp_score


def gmp_score(uncertainty_model, X_test, y_test, significance): 
    
    if uncertainty_model.__class__.__name__ == "BayesianBootstrapRegressor": 
        y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test)    
        
    if uncertainty_model.__class__.__name__ == "AggregatedCp": 
        intervals = uncertainty_model.predict(X_test, significance = 0.3173)
        y_mu = [np.mean(interval) for interval in intervals]
        y_sigma = [(interval[1]-interval[0])/2 for interval in intervals]

    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test, n_samples = 1000)   
        
    probs = st.norm.pdf(y_test, loc = y_mu, scale = y_sigma)
    gmp_score = st.mstats.gmean(probs)

    return gmp_score

#%%

# relationship between uncertainty and applicability domain
# data sparsity / neighbour density represents applicability domain

def data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    # Data sparsity metric based on KNN
    
    if uncertainty_model.__class__.__name__ == "AggregatedCp": 
        intervals = uncertainty_model.predict(X_test, significance = 0.3173)
        y_mu = [np.mean(interval) for interval in intervals]
        y_sigma = [(interval[1]-interval[0])/2 for interval in intervals]
    
    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        y_mu, y_sigma = uncertainty_model.predict_normal_distr(X_test, n_samples=1000)



    knn = KNeighborsRegressor(n_neighbors = n_neighbors, weights = 'distance', metric = distance)
    knn.fit(X_train, y_train)
    dists_list = []
    
    for mol in X_test: 
        dists = knn.kneighbors(mol.reshape(1,-1))[0][0]
        mean_dist = np.sum(dists)/(n_neighbors)
        dists_list.append(mean_dist)
    
    return y_sigma, dists_list

def data_sparsity_plot(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    y_sigma, dists_list = data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    graph_max = np.amax(np.maximum(y_sigma, dists_list))
    plt.scatter(dists_list, y_sigma, s = 2)
    #plt.plot([0,graph_max],[0,graph_max], c = 'k', lw=1)
    plt.xlabel("Mean distance to %s nearest neighbors" % n_neighbors)
    plt.ylabel("Size of total uncertainty")  

    plt.show()
    
    return

def data_sparsity_r2_score(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    y_sigma, dists_list = data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    score = r2_score(y_sigma, dists_list)
    
    return score
    
def data_sparsity_pearsonr(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 

    y_sigma, dists_list = data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    score, p_value = st.pearsonr(y_sigma, dists_list)
    
    return score


def epis_data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    # Data sparsity metric based on KNN
    # Distance metric: 'jaccard' for fingerprints, 'euclidean' for tSVD / continuous representations
    # Epis from bayesianNN
    
    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        epis, epis_mu, epis_sigma = uncertainty_model.get_epistemic(X_test, n_samples=1000)


    knn = KNeighborsRegressor(n_neighbors = n_neighbors, weights = 'distance', metric = distance)
    knn.fit(X_train, y_train)
    dists_list = []
    
    for mol in X_test: 
        dists = knn.kneighbors(mol.reshape(1,-1))[0][0]
        mean_dist = np.sum(dists)/(n_neighbors)
        dists_list.append(mean_dist)
    
    return epis, dists_list

def epis_data_sparsity_plot(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    epis, dists_list = epis_data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    graph_max = np.amax(np.maximum(epis, dists_list))
    plt.scatter(dists_list, epis, s = 2)
    #plt.plot([0,graph_max],[0,graph_max], c = 'k', lw=1)
    plt.xlabel("Mean distance to %s nearest neighbors" % n_neighbors)
    plt.ylabel("Size of epistemic uncertainty")  

    plt.show()
    
    return

def epis_data_sparsity_r2_score(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    epis, dists_list = epis_data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    score = r2_score(epis, dists_list)
    
    return score

def epis_data_sparsity_pearsonr(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 

    epis, dists_list = epis_data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    score, p_value = st.pearsonr(epis, dists_list)
    
    return score

def alea_data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    # Data sparsity metric based on KNN
    # Distance metric: 'jaccard' for fingerprints, 'euclidean' for tSVD / continuous representations
    # Alea from bayesianNN
    
    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        alea, alea_mu, alea_sigma = uncertainty_model.get_aleatoric(X_test, n_samples=1000)


    knn = KNeighborsRegressor(n_neighbors = n_neighbors, weights = 'distance', metric = distance)
    knn.fit(X_train, y_train)
    dists_list = []
    
    for mol in X_test: 
        dists = knn.kneighbors(mol.reshape(1,-1))[0][0]
        mean_dist = np.sum(dists)/(n_neighbors)
        dists_list.append(mean_dist)
    
    return alea, dists_list


def alea_data_sparsity_plot(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    alea, dists_list = alea_data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    graph_max = np.amax(np.maximum(alea, dists_list))
    plt.scatter(dists_list, alea, s = 2)
    plt.plot([0,graph_max],[0,graph_max], c = 'k', lw=1)
    plt.xlabel("Mean distance to %s nearest neighbors" % n_neighbors)
    plt.ylabel("Size of aleatoric uncertainty")  

    plt.show()
    
    return

def alea_data_sparsity_r2_score(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    alea, dists_list = alea_data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    score = r2_score(alea, dists_list)
    
    return score

def alea_data_sparsity_pearsonr(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 

    alea, dists_list = alea_data_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    score, p_value = st.pearsonr(alea, dists_list)
    
    return score

#%%

# latent space as a notion of applicability domain (didn't really work)

def epis_latent_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    # Data sparsity metric based on KNN
    # Latent space distance calculated from neural network's last hidden layer
    # Distance metric: 'jaccard' for fingerprints, 'euclidean' for tSVD / continuous representations
    # Epis from bayesianNN
    
    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        epis, epis_mu, epis_sigma = uncertainty_model.get_epistemic(X_test, n_samples=1000)
        X_train_lat = uncertainty_model.get_latent(X_train, n_samples=500)
        X_test_lat = uncertainty_model.get_latent(X_test, n_samples=500)

    knn = KNeighborsRegressor(n_neighbors = n_neighbors, weights = 'distance', metric = distance)
    knn.fit(X_train_lat, y_train)
    dists_list = []
    
    for mol in X_test_lat: 
        dists = knn.kneighbors(mol.reshape(1,-1))[0][0]
        mean_dist = np.sum(dists)/(n_neighbors)
        dists_list.append(mean_dist)
    
    return epis, dists_list

def epis_latent_sparsity_plot(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    epis, dists_list = epis_latent_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    graph_max = np.amax(np.maximum(epis, dists_list))
    plt.scatter(dists_list, epis, s = 2)
    #plt.plot([0,graph_max],[0,graph_max], c = 'k', lw=1)
    plt.xlabel("Mean distance to %s nearest neighbors" % n_neighbors)
    plt.ylabel("Size of epistemic uncertainty")  

    plt.show()
    
    return

def epis_latent_sparsity_r2_score(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 
    
    epis, dists_list = epis_latent_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    score = r2_score(epis, dists_list)
    
    return score

def epis_latent_sparsity_pearsonr(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance): 

    epis, dists_list = epis_latent_sparsity(uncertainty_model, X_train, y_train, X_test, n_neighbors, distance)
    score, p_value = st.pearsonr(epis, dists_list)
    
    return score
    
#%%

# other experimental ideas that did not work

def absolute_error(uncertainty_model, X_test, y_test, significance): 
    
    # Absolute error = np.abs(y_pred - y_test)

    if uncertainty_model.__class__.__name__ == "AggregatedCp": 
        intervals = uncertainty_model.predict(X_test, significance = significance)    
    
    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        intervals = uncertainty_model.predict_central_interval(X_test, n_samples=1000, alpha = significance)

    interval_size = intervals[:,1] - intervals[:,0]    
    y_pred = np.mean(intervals[:,0],intervals[:,1])
    y_err = np.abs(y_pred - y_test)

    return interval_size, y_err


def absolute_error_plot(uncertainty_model, X_test, y_test, significance): 
    
    interval_size, y_err_list = mean_deviation(uncertainty_model, X_test, y_test, significance)
    
    plt.scatter(y_err_list, interval_size, s = 2)
    plt.plot([0,3],[0,3], c = 'k', lw=1)
    plt.xlabel("Mean absolute error")
    plt.ylabel("Size of confidence interval")  
    plt.show()

    return

def absolute_error_r2_Score(uncertainty_model, X_test, y_test, significance): 
    
    interval_size, y_err_list = mean_deviation(uncertainty_model, X_test, y_test, significance)
    score = r2_score(interval_size, y_err_list)

    return score


def mean_deviation(uncertainty_model, X_test, y_test, significance): 
    
    # Deviation from mean = np.abs(y_pred - y_mean)
    
    if uncertainty_model.__class__.__name__ == "AggregatedCp": 
        intervals = uncertainty_model.predict(X_test, significance = significance)    
    
    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        intervals = uncertainty_model.predict_central_interval(X_test, n_samples=1000, alpha = significance)

    interval_size = intervals[:,1] - intervals[:,0]    
    y_pred = [np.mean(i) for i in intervals]
    y_mean = np.mean(y_test)
    y_dev = np.abs(y_pred - y_mean)

    return interval_size, y_dev

def mean_deviation_plot(uncertainty_model, X_test, y_test, significance): 
    
    interval_size, y_dev_list = mean_deviation(uncertainty_model, X_test, y_test, significance)
    
    plt.scatter(y_dev_list, interval_size, s = 2)
    plt.plot([0,3],[0,3], c = 'k', lw=1)
    plt.xlabel("Deviation from dataset mean")
    plt.ylabel("Size of confidence interval")  
    plt.show()

    return

def mean_deviation_r2_score(uncertainty_model, X_test, y_test, significance):
    
    interval_size, y_dev_list = mean_deviation(uncertainty_model, X_test, y_test, significance)
    score = r2_score(interval_size, y_dev_list)
    
    return score
    
def ll_latent_coords(uncertainty_model, X_test, y_test, significance, n_neighbors): 
    
    # ONLY WORKS FOR NEURAL NETWORKS
    
    if uncertainty_model.__class__.__name__ == "BayesianNN": 
        intermediate_model = tf.keras.Model(uncertainty_model.input, uncertainty_model.get_layer(index=-3).output)
        intervals = uncertainty_model.predict_central_interval(X_test, n_samples=1000, alpha = significance)
        
    interval_size = intervals[:,1] - intervals[:,0]
    latent = intermediate_model(X_test).numpy()
    pca = PCA(n_components = 3)
    translatent = pca.fit_transform(latent)
    
    knn = KNeighborsRegressor(n_neighbors = n_neighbors)
    knn.fit(translatent, y_test)
    dists_list = []
    
    for mol in translatent: 
        dists = knn.kneighbors(mol.reshape(1,-1))[0][0]
        mean_dist = np.sum(dists[1:-1])/(n_neighbors-1)
        dists_list.append(mean_dist)
    
    return interval_size, dists_list

def ll_latent_plot(uncertainty_model, X_test, y_test, significance, n_neighbors): 
    
    interval_size, dists_list = ll_latent_coords(uncertainty_model, X_test, y_test, significance, n_neighbors)
    graph_max = np.amax(np.maximum(interval_size, dists_list))
    plt.scatter(dists_list, interval_size, s = 2)
    plt.plot([0,graph_max],[0,graph_max], c = 'k', lw=1)
    plt.xlabel("Mean distance to %s nearest neighbors" % n_neighbors)
    plt.ylabel("Size of confidence interval")  

    plt.show()
    
    return

def ll_latent_r2_score(uncertainty_model, X_test, y_test, significance, n_neighbors): 
    
    interval_size, dists_list = ll_latent_coords(uncertainty_model, X_test, y_test, significance, n_neighbors)
    score = r2_score(interval_size, dists_list)
    
    return score