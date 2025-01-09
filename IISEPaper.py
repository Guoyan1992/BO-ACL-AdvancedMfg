from operator import index
import copy
import pandas as pd 
import numpy as np
from doepy import build
import seaborn as sns
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from clusterMethod import get_clustering_prediction 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from sklearn.cluster import SpectralClustering, AffinityPropagation
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.gaussian_process.kernels import WhiteKernel

def categorise(row):  
    if 1.6 >= row['x1'] >= 0.3 and 0.3 <= row['x2'] <= 1.6:
        return 1
    elif -0.3 >= row['x1'] >= -1.6 and -1.6 <= row['x2'] <= -0.3:
        return 1
    elif 1.6 >= row['x1'] >= 0.3 and -1.6 <= row['x2'] <= -0.3:
        return 1
    elif -0.3 >= row['x1'] >= -1.6 and 0.3 <= row['x2'] <= 1.6:
        return 1

def get_labeled_index(whole, initial):
    arr1 = np.array(whole.iloc[:,:2])
    arr2 = np.array(initial.iloc[:,:2])
    index = []
    for i in np.arange(arr2.shape[0]):
        ind = np.where(np.all(arr1==arr2[i],axis=1))[0][0]
        index.append(ind)
    index = list(set(index))
    return index

def get_div_term(unlabeled_feature,labeled_feature):
    dis_div = euclidean_distances(unlabeled_feature,np.array(labeled_feature))
    div = dis_div.min(axis=1)
    return div

def get_rep_term(unlabeled_feature,i):
    y_Kmean,y_centriods = get_clustering_prediction(i,unlabeled_feature)
    dis_rep = euclidean_distances(unlabeled_feature,y_centriods[0].reshape(1,-1))
    dis_rep = 1- (dis_rep - dis_rep.min())/(dis_rep.max()-dis_rep.min())
    return dis_rep

def get_select_index(X_feature, select_sample):
    select_index=[]
    for i in np.arange(select_sample.shape[0]):
        ind = np.where(np.all(X_feature==np.array(select_sample)[i,:],axis=1))[0][0]
        select_index.append(ind)
    return select_index

def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

def get_labeled_feasible_sample(train_set, initial_feasible_region):
    ind = train_set.loc[initial_feasible_region]
    initial_feasible_sample = np.array(ind)
    return initial_feasible_sample

def get_feasible_rep(c_pool,initial_feasible_sample):
    feasible_rep = euclidean_distances(np.array(c_pool.iloc[:,:2]),initial_feasible_sample)
    rep_new = feasible_rep.mean(axis=1)
    rep_new = 1- (rep_new - rep_new.min())/(rep_new.max()-rep_new.min())
    return rep_new

def c_prediction(train_s_set, labeled_y_c, unlabeled_data,model):
    model.fit(train_s_set, labeled_y_c)
    y_c_pred = model.predict(unlabeled_data)
    y_prob = model.predict_proba(unlabeled_data)
    C_uncertainty = 1-abs(y_prob[:,0]-y_prob[:,1])
    C_uncertainty = (C_uncertainty - C_uncertainty.min())/(C_uncertainty.max()-C_uncertainty.min())
    return y_c_pred, y_prob,C_uncertainty

def get_raw_rep(c_pool,initial_feasible_sample):
    feasible_rep = euclidean_distances(np.array(c_pool.iloc[:,:2]),initial_feasible_sample)
    rep_new = feasible_rep.mean(axis=1)
    return rep_new

def get_raw_div(unlabeled_feature,labeled_feature):
    dis_div = euclidean_distances(unlabeled_feature,np.array(labeled_feature))
    div = dis_div.min(axis=1)
    return div

def get_feature_label(dataset):
    feature_name = list(dataset.columns)[:-2]
    objective_name_r = list(dataset.columns)[-2]
    objective_name_c = list(dataset.columns)[-1]
    ds = copy.deepcopy(dataset) 
    X_feature = ds[feature_name].values
    y_c = np.array(ds[objective_name_c].values)
    y_r = np.array(ds[objective_name_r].values)
    return ds, X_feature, y_c, y_r

def get_initial_selection(dataset, initialsize,ran_list,i):
    indices = list(np.arange(len(dataset)))
    index_total = indices.copy()
    index_learn=index_total.copy()
    random.seed(ran_list[i])
    index_select = random.sample(index_learn, initialsize)
    return index_select, index_learn
    
    
def get_labeled_set(selected_index,index,X_feature, y_c,y_r):
    X_labeled= []
    #     list to store all observed good candidates' objective value y
    y_c_labeled= []
    y_r_labeled= []
    for i in selected_index:
        X_labeled.append(X_feature[i])
        y_c_labeled.append(y_c[i])
        y_r_labeled.append(y_r[i])
        index.remove(i) 
    return X_labeled, y_c_labeled, y_r_labeled, index

def get_training_Set(X_labeled, y_c_labeled, y_r_labeled):
    training_data = pd.DataFrame(np.c_[np.array(X_labeled),y_c_labeled,y_r_labeled],columns=['x1','x2','c','r'])
    train_c_set = training_data.iloc[:,:2]
    train_r_set ,r_labeled= training_data[training_data['c']==0] .iloc[:,:2],training_data[training_data['c']==0] .iloc[:,-1]
    return training_data, train_c_set, train_r_set, r_labeled
    
def get_unlabeled_set(index_learn, X_feature, y_c, y_r):
    X_pool=[]
    y_c_pool=[]
    y_r_pool=[]
    for j in index_learn:
            X_pool.append(X_feature[j])
            y_c_pool.append(y_c[j])
            y_r_pool.append(y_r[j])
    return X_pool, y_c_pool, y_r_pool

def get_unselection_index(selected_index,X_feature):
    index = list(np.arange(X_feature.shape[0]))
    for i in selected_index:
        index.remove(i) 
    return index

def predicted_region_divide(unlabeled_data,y_c_pool,y_r_pool,y_c_pred,y_c_uncertainty):
    un_pool = pd.DataFrame(np.c_[np.array(unlabeled_data),y_c_pool,y_r_pool,y_c_pred,y_c_uncertainty],columns=['x1','x2','c','r','c_pred','c_uncertain'])
    # predicted unfeasible region 
    temp_c = un_pool[un_pool['c_pred'] == 1]
    # predicted feasible region
    temp_r = un_pool[un_pool['c_pred'] == 0]
    return un_pool, temp_c, temp_r



def GP_predict(X_train,Y_train,X_pool):
    kernel = 2.0 * Matern(length_scale=(2,), nu=2.5)+ WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,alpha=0.1, normalize_y=True,optimizer='fmin_l_bfgs_b')
    gp.fit(X_train,Y_train)
    y_pred, sigma = gp.predict(X_pool, return_std=True)
    return y_pred,sigma

def labeled_fea_sample(x_labeled,y_labeled,y_r_labeled):
    index_0 = np.where(y_labeled == 0)[0]
    index_1 = np.where(y_labeled == 1)[0]
    return x_labeled[index_0], x_labeled[index_1],y_r_labeled[index_0]

def remove_label(full_list,remove_list):
    for n in remove_list:
        while n in full_list:
            full_list.remove(n)
            
    return full_list

def get_index_div(div, unlabeled_index):
    index=[]
    max_div = max(div)
    temp = list(div).index(max_div)
    index.append(unlabeled_index[temp])
    return index
