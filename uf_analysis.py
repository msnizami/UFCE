#!/usr/bin/env python
# coding: utf-8

# #### Import libraries

import os
import glob
import time
import gower
import random
import pandas as pd
import numpy as np
import seaborn as sns
from numpy import arange
from numpy import hstack
from scipy import stats
from numpy import meshgrid
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from IPython.display import display
from pandas.errors import EmptyDataError
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_absolute_deviation
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression

plt.style.use("seaborn-whitegrid")
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000

import ufce
from ufce import UFCE
from goodness import *
from cfmethods import *
from evaluations import *
from data_processing import *
from generate_text_explanations import *

ufc = UFCE()

# this file contain the experiment code for user-feedback analysis with different levels of user-constraints, specificaly customised for Bank Loan dataset.

def compute_percentage_for_uf_analysis(df, f2c,fcat, seed):
    # Compute the median absolute deviation of each feature
    mad = df.std()
    # Add 5% of the MAD to each feature and store the updated values in a dictionary
    updated_values = {}
    list_update_values = []
    for feature in f2c:
        if feature == 'Education':
            updated_values[feature] = 1
        else:
            if feature == 'CCAvg':
                updated_values[feature] = round(mad[feature]*seed, 2)
            else:
                updated_values[feature] = round(mad[feature] * seed)
    for feature in fcat:
        updated_values[feature] = 1
    list_update_values.append(updated_values)
    return list_update_values

def modify_testinstance(test, cf, uf):
    temp = test.copy()
    for feature in uf.keys():
        temp[feature] = test[feature].values[0] + uf[feature]
    for f in cf:
        if temp[f].values < cf[f].values or cf[f].values < test[f].values:
            return False
        return True


# #### Read data
#path = r'/home/m.suffian/Downloads/UFCE-4GPU/data/'  # use this path on ubuntu. make sure you have correct path to UFCE folder.
path = r'C:\Users\~\UFCE-4GPU\data' # use this path format on windows system, verify your drive path to UFCE
pathbank = r'C:\Users\~\UFCE-4GPU\data\bank.csv'
datasetdf = pd.read_csv(pathbank)
datasetdf = datasetdf.sample(frac=1)
# print(datasetdf.mad(), datasetdf.std())

mlp, mlp_mean, mlp_std, lr, lr_mean, lr_std, Xtest, Xtrain, X, Y, df = classify_dataset_getModel(datasetdf[:200], data_name='bank') # this method returns trained ML model's, cleaned dataframe, and etc.
models = {lr: [lr_mean, lr_std]} #, mlp: [mlp_mean, mlp_std]}

print(f'Bank Dataset')

readpath = r'C:\Users\~\UFCE-4GPU\folds\bank\totest\testfold_0_pred_0.csv'
writepath = r'C:\Users\~\UFCE-4GPU\folds\bank\totest\\'
features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1 = get_bank_user_constraints(df) # this will return user-constraints specific to data set.
# del data_lab1['Unnamed: 0']
# del data_lab1['age']
# del data_lab1['Experience']

# Take top mutual information sharing pairs of features
MI_FP = ufc.get_top_MI_features(X, features)
print(f'\t Top-5 Mutually-informed feature paris:{MI_FP[:5]}')

f2change = ['Income', 'CCAvg', 'Mortgage', 'Education']
f2cat = ['CDAccount', 'Online']

# different levels of values (specific percentage of MAD represents to a diferent user value)
uf1 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.20) # 20% of mad
uf2 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.40) # 40% of mad
uf3 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.60) # 60% of mad
uf4 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 0.80) # 80% of mad
uf5 = compute_percentage_for_uf_analysis(df, f2change, f2cat, 1.0) # # 100% of mad
ufs = [uf1, uf2, uf3, uf4, uf5]
print(ufs)
models = ['lr'] # the experiment is run for 'lr' (Logistic Regression) as AR method doesn't work on 'mlp'

mnames = ['ufce1','ufce2', 'ufce3', 'dice', 'ar']
percent_cfs_all = pd.DataFrame()
time_cfs_all = pd.DataFrame()
for u, uf in enumerate(ufs):
    print(f' user feedback{u}:', uf[0])
    cfmethods = ['UFCE1', 'UFCE2', 'UFCE3', 'DiCE', 'AR']
    methodtimes = dict()
    cfcounts = dict()
    no_cf = 1
    k = 1
    bb = lr
    desired_outcome = desired_outcome
    protectf = protectf
    meandf_catproximity = pd.DataFrame(columns=mnames)
    stddf_catproximity = pd.DataFrame(columns=mnames)
    meandf_contproximity = pd.DataFrame(columns=mnames)
    stddf_contproximity = pd.DataFrame(columns=mnames)
    meandf_sparsity = pd.DataFrame(columns=mnames)
    stddf_sparsity = pd.DataFrame(columns=mnames)
    meandf_actionability = pd.DataFrame(columns=mnames)
    stddf_actionability = pd.DataFrame(columns=mnames)
    meandf_plausibility = pd.DataFrame(columns=mnames)
    stddf_plausibility = pd.DataFrame(columns=mnames)
    meandf_feasibility = pd.DataFrame(columns=mnames)
    stddf_feasibility = pd.DataFrame(columns=mnames)
    meandf_diversity = pd.DataFrame(columns=mnames)
    stddf_diversity = pd.DataFrame(columns=mnames)
    meandf_feasibility = pd.DataFrame(columns=mnames)
    stddf_feasibility = pd.DataFrame(columns=mnames)
    cols = ['ufce1mean', 'ufce2mean', 'ufce3mean','dicemean', 'armean',  'ufce1std', 'ufce2std',
            'ufce3std', 'dicestd', 'arstd']
    jproxidf = pd.DataFrame(columns=cols)
    catproxidf = pd.DataFrame(columns=cols)
    contproxidf = pd.DataFrame(columns=cols)
    spardf = pd.DataFrame(columns=cols)
    actdf = pd.DataFrame(columns=cols)
    plausdf = pd.DataFrame(columns=cols)
    feasidf = pd.DataFrame(columns=cols)
    # for i, file in enumerate(testfolds[:]):
    #     print(f'\t\t\t Test Fold:{i} ')
    #     try:
    #         testset = pd.read_csv(file)
    #     except pd.io.common.EmptyDataError:
    #         print('File is empty or not found')
    #     else:
    testset = pd.read_csv(readpath)
    for i, method in enumerate(cfmethods):
        print(f'\t\t\t\t Method: {method}  --------------')
        if method == 'DiCE':
            dicecfs, methodtimes[i] = dice_cfexp(df, data_lab1, uf[0], MI_FP[:5], testset[:], numf, f2change, outcome_label, k, bb)
            del dicecfs[outcome_label]
            if len(dicecfs) != 0:
                count = 0
                for x in range(len(testset[:])):
                    flag = modify_testinstance(testset[x:x+1], dicecfs[x:x+1], uf[0])
                    if flag:
                        count = count + 1
                cfcounts[i] = count/len(testset[:]) * 100
            else:
                cfcounts[i] = 0
            #print(f'\t\t\t\t Counterfactuals \t:{dicecfs.values}')
        elif method == 'AR':
            arcfs, methodtimes[i] = ar_cfexp(X, numf, bb, testset[:])
            if len(arcfs) != 0:
                count = 0
                for x in range(len(testset[:])):
                    flag = modify_testinstance(testset[x:x + 1], arcfs[x:x + 1], uf[0])
                    if flag:
                        count = count + 1
                cfcounts[i] = count / len(testset[:]) * 100
            else:
                cfcounts[i] = 0
            #print(f'\t\t\t\t Counterfactual \t:{arcfs.values}')
        elif method == 'UFCE1':
            onecfs, methodtimes[i], foundidx1, interval1, testout1 = sfexp(X, data_lab1, testset[:], uf[0], step, f2change, numf, catf, bb, desired_outcome, k)
            if len(onecfs) != 0:
                cfcounts[i] = len(onecfs)/len(testset[:]) * 100
            else:
                cfcounts[i] = 0
            # for id in foundidx1:
            #     print(f'\t\t\t\t{id} Test instance \t:{testset[id:id+1].values}')
            #     print(f'\t\t\t\t UF with MC \t:{interval1[id]}')
            #     print(f'\t\t Counterfactual \t:{onecfs[id:id+1].values}')
        elif method == 'UFCE2':
            twocfs, methodtimes[i], foundidx2, interval2, testout2 = dfexp(X, data_lab1, testset[:], uf[0], MI_FP[:5], numf, catf, features, protectf, bb, desired_outcome, k)
            if len(twocfs) != 0:
                cfcounts[i] = len(twocfs)/len(testset[:]) * 100
            else:
                cfcounts[i] = 0
            # for id in foundidx2:
            #     print(f'\t\t\t\t{id} Test instance \t:{testset[id:id + 1].values}')
            #     print(f'\t\t\t\t UF with MC \t:{interval2[id]}')
            #     print(f'\t\t\t\t Counterfactual \t:{twocfs[id:id + 1].values}')
        else:
            threecfs, methodtimes[i], foundidx3, interval3, testout3 = tfexp(X, data_lab1, testset[:], uf[0], MI_FP[:5], numf, catf, features, protectf, bb, desired_outcome, k)
            if len(threecfs) != 0:
                cfcounts[i] = len(threecfs)/len(testset[:]) * 100
            else:
                cfcounts[i] = 0
            # for id in foundidx3:
            #     print(f'\t\t{id} Test instance \t:{testset[id:id + 1].values}')
            #     print(f'\t\t UF with MC \t:{interval3[id]}')
            #     print(f'\t\t Counterfactual \t:{threecfs[id:id + 1].values}')

    # calling all 7 evaluation metrics (properties)
    # # categorical proximity
    mmeans, mstds = [], []
    mmeans, mstds = Catproximity(onecfs, testout1, twocfs, testout2, threecfs, testout3, dicecfs, arcfs, testset, catf)
    df = pd.DataFrame(data=[mmeans], columns=mnames)
    meandf_catproximity = pd.concat([meandf_catproximity, df], ignore_index=True, axis=0)
    df = pd.DataFrame(data=[mstds], columns=mnames)
    stddf_catproximity = pd.concat([stddf_catproximity, df], ignore_index=True, axis=0)
    mmeans.extend(mstds)
    df = pd.DataFrame(data=[mmeans], columns=cols)
    catproxidf = pd.concat([catproxidf, df], ignore_index=True, axis=0)
    # continuous proximity
    mmeans, mstds = [], []
    mmeans, mstds = Contproximity(onecfs, testout1, twocfs, testout2, threecfs, testout3, dicecfs, arcfs, Xtest, numf)
    df = pd.DataFrame(data=[mmeans], columns=mnames)
    meandf_contproximity = pd.concat([meandf_contproximity, df], ignore_index=True, axis=0)
    df = pd.DataFrame(data=[mstds], columns=mnames)
    stddf_contproximity = pd.concat([stddf_contproximity, df], ignore_index=True, axis=0)
    mmeans.extend(mstds)
    df = pd.DataFrame(data=[mmeans], columns=cols)
    contproxidf = pd.concat([contproxidf, df], ignore_index=True, axis=0)
    # sparsity
    mmeans, mstds = [], []
    mmeans, mstds = Sparsity(onecfs, testout1, twocfs, testout2, threecfs, testout3, dicecfs, arcfs, Xtest, numf)
    df = pd.DataFrame(data=[mmeans], columns=mnames)
    meandf_sparsity = pd.concat([meandf_sparsity, df], ignore_index=True, axis=0)
    df = pd.DataFrame(data=[mstds], columns=mnames)
    stddf_sparsity = pd.concat([stddf_sparsity, df], ignore_index=True, axis=0)
    mmeans.extend(mstds)
    df = pd.DataFrame(data=[mmeans], columns=cols)
    spardf = pd.concat([spardf, df], ignore_index=True, axis=0)
    # actionability
    mmeans, mstds = [], []
    mmeans, mstds = Actionability(onecfs, testout1, twocfs, testout2, threecfs, testout3, dicecfs, arcfs, Xtest, features, f2change)
    df = pd.DataFrame(data=[mmeans], columns=mnames)
    meandf_actionability = pd.concat([meandf_actionability, df], ignore_index=True, axis=0)
    df = pd.DataFrame(data=[mstds], columns=mnames)
    stddf_actionability = pd.concat([stddf_actionability, df], ignore_index=True, axis=0)
    mmeans.extend(mstds)
    df = pd.DataFrame(data=[mmeans], columns=cols)
    actdf = pd.concat([actdf, df], ignore_index=True, axis=0)
    mmeans, mstds = [], []
    mmeans, mstds = Plausibility(onecfs, twocfs, threecfs, dicecfs, arcfs, Xtest, Xtrain)
    df = pd.DataFrame(data=[mmeans], columns=mnames)
    meandf_plausibility = pd.concat([meandf_plausibility, df], ignore_index=True, axis=0)
    df = pd.DataFrame(data=[mstds], columns=mnames)
    stddf_plausibility = pd.concat([stddf_plausibility, df], ignore_index=True, axis=0)
    mmeans.extend(mstds)
    df = pd.DataFrame(data=[mmeans], columns=cols)
    plausdf = pd.concat([plausdf, df], ignore_index=True, axis=0)
    mmeans, mstds = [], []
    mmeans, mstds = Feasibility(onecfs, twocfs, threecfs, dicecfs, arcfs, Xtest, Xtrain, features, f2change, bb,
                         desired_outcome, outcome_label)
    df = pd.DataFrame(data=[mmeans], columns=mnames)
    meandf_feasibility = pd.concat([meandf_feasibility, df], ignore_index=True, axis=0)
    df = pd.DataFrame(data=[mstds], columns=mnames)
    stddf_feasibility = pd.concat([stddf_feasibility, df], ignore_index=True, axis=0)
    mmeans.extend(mstds)
    df = pd.DataFrame(data=[mmeans], columns=cols)
    feasidf = pd.concat([feasidf, df], ignore_index=True, axis=0)

    # here storing the time and percentage of counterfactuals for each cfmethod.
    temptime = pd.DataFrame([methodtimes])
    time_cfs_all = pd.concat([time_cfs_all, temptime], ignore_index=True)
    tempcount = pd.DataFrame([cfcounts])
    percent_cfs_all = pd.concat([percent_cfs_all, tempcount], ignore_index=True)
print(display(time_cfs_all))
print(display(percent_cfs_all))
print(f'\t\t\t\t-----fold_mean_values of all evaluation metrics----')

print(f'Mean and St.dev of Joint-Proximity:', jproxidf.to_latex(float_format="{:0.2f}".format))
print(f'Mean and St.dev of Cat-Proximity:', catproxidf.to_latex(float_format="{:0.2f}".format))
print(f'Mean and St.dev of Cont-Proximity:', contproxidf.to_latex(float_format="{:0.2f}".format))
print(f'Mean and St.dev of Sparsity:', spardf.to_latex(float_format="{:0.2f}".format))
print(f'Mean and St.dev of Actionability:', actdf.to_latex(float_format="{:0.2f}".format))
print(f'Mean and St.dev of Plausibility:', plausdf.to_latex(float_format="{:0.2f}".format))
print(f'Mean and St.dev of Feasibility:', feasidf.to_latex(float_format="{:0.2f}".format))

