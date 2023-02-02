# -*- coding: utf-8 -*-
"""
Created on Thursday Feb 24 17:44:07 2022

@author: muhammad suffian
"""

import pandas as pd
import numpy as np
from ufce import UFCE
from bank_ufce import Bank_UFCE
from datasets import DATSETS
from ufce_spotify import UFCE_Spotify
from goodness import *
from data_processing import *
from dummy_scaler import DummyScaler
import pickle
#from ufce_buildrules import BuildTrees
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import sklearn.neural_network as sknet


ufc = UFCE()
ufcbank = Bank_UFCE()
dataset = DATSETS()
#ufcspotify = UFCE_Spotify()



# lg models
lg_bank, lg_credit, lg_adult, lg_append, lg_bupa, lg_heart, lg_movie, lg_mammo, lg_saheart, lg_spotify, lg_titanic, lg_wdbc, lg_wine, lg_wiscon = get_model()

# mlp models
mlp_bank, mlp_credit, mlp_adult, mlp_append, mlp_bupa, mlp_heart, mlp_movie, mlp_mammo, mlp_saheart, mlp_spotify, mlp_titanic, mlp_wdbc, mlp_wine, mlp_wiscon = get_model_mlp()


#Xtrain data
Xtrain = process_data_get_Xtrain()
#Xtrain.columns = ['At1', 'At2', 'At3', 'At4', 'At5', 'At6', 'At7']
Xtrain.drop(columns=Xtrain.columns[0], axis=1, inplace=True)
#Xtrain.drop(columns=Xtrain.columns[0], axis=1, inplace=True)
#Xtrain.drop(columns=Xtrain.columns[1], axis=1, inplace=True)
#Xtrain.drop(columns=Xtrain.columns[2], axis=1, inplace=True)

print(Xtrain.columns)

#Xtest data
bank, credit, adult, append, bupa, heart, magic, mammo, movie, saheart, spotify, titanic, wdbc, wine, wiscon = process_data_get_Xtest()
datasets_data = [bank, credit, adult, append, bupa, heart, mammo, movie, saheart, spotify, titanic, wdbc, wine, wiscon]
#taking k=100 xtest instances
bank = bank.sample(frac=1)
Xtest_bank = bank[:100] if bank.shape[0] >=100 else bank[:]
credit = credit.sample(frac=1)
Xtest_credit = credit[:100] if credit.shape[0] >=100 else credit[:]
adult = adult.sample(frac=1)
Xtest_adult = adult[:100] if adult.shape[0] >=100 else adult[:]
append = append.sample(frac=1)
Xtest_append = append[:100] if append.shape[0] >=100 else append[:]
bupa = bupa.sample(frac=1)
Xtest_bupa = bupa[:100] if bupa.shape[0] >=100 else bupa[:]
heart = heart.sample(frac=1)
Xtest_heart = heart[:100] if heart.shape[0] >=100 else heart[:]
#mammo = mammo.sample(frac=1)
Xtest_mammo = mammo[:100] if mammo.shape[0] >=100 else mammo[:]
magic = magic.sample(frac=1)
Xtest_magic = magic[:100] if magic.shape[0] >=100 else magic[:]
#movie = movie.sample(frac=1)
Xtest_movie = movie[:100] if movie.shape[0] >=100 else movie[:]
#saheart = saheart.sample(frac=1)
Xtest_saheart = saheart[:100] if saheart.shape[0] >=100 else saheart[:]
#spotify = spotify.sample(frac=1)
Xtest_spotify = spotify[:100] if spotify.shape[0] >=100 else spotify[:]
#titanic = titanic.sample(frac=1)
Xtest_titanic = titanic[:100] if titanic.shape[0] >=100 else titanic[:]
#wdbc = wdbc.sample(frac=1)
Xtest_wdbc = wdbc[:100] if wdbc.shape[0] >=100 else wdbc[:]
#wine = wine.sample(frac=1)
Xtest_wine = wine[:100] if wine.shape[0] >=100 else wine[:]
wiscon = wiscon.sample(frac=1)
Xtest_wiscon = wiscon[:100] if wiscon.shape[0] >=100 else wiscon[:]

Xtest = Xtest_wiscon
del Xtest['Unnamed: 0']
#del Xtest['Unnamed: 0.1']
#del Xtest_heart['Age']
#del Xtest_heart['Sex']
y_val = mlp_wiscon.predict(Xtest[:20])
print(y_val)

scaler = DummyScaler()
scaler.fit(Xtrain)

one_path = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v3\\wisconsin\\mlp\\one\\'
two_path = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v3\\wisconsin\\mlp\\two\\'
three_path = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v3\\wisconsin\\mlp\\three\\'
K = 20

validity_cf_1f = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
validity_cf_2f = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
validity_cf_3f = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
desired_outcome = 2.0
features, user_feature_list, feature_flags, threshold_values, order_of_asymmetric, protected_features, features_for_corr, perturbing_rates, u_f_cat_list = dataset.wisconsin_details()
c = 0
del wiscon['Unnamed: 0']
#adult.columns = 'At1', 'At2', 'At3', 'At4', 'At5', 'At6', 'At7']
print(wiscon.columns)
#Xtest_adult.columns = ['workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','hours-per-week','native-country','net_capital','per-hour-income']
#del Xtest_bupa['Unnamed: 0']
print(Xtest.columns)
Xtest = Xtest.sort_values(by=Xtest.columns[0], ascending=False)
K = 20
'''
# Generation of Counterfactuals for all 3-formats (20 Cfs)
for k in range(K):
    cfcount1 = 0
    cfcount2 = 0
    cfcount3 = 0
    for t in range(len(Xtest[:K])):
        intervals = ufc.user_feedback_processing(Xtest[t:t+1], user_feature_list, feature_flags, threshold_values, order_of_asymmetric)
        #print("here", Xtest_append[t:t+1])
        intervals = ufc.make_interval_faithful(intervals, 'wisconsin')
        one = ufc.one_feature_synthetic_data_testing(Xtest[t:t + 1], u_f_cat_list, intervals, protected_features, perturbing_rates, mlp_wiscon, desired_outcome, k+1)
        two = ufc.two_feature_synthetic_data(wiscon, Xtest[t:t + 1], features_for_corr, u_f_cat_list, intervals,features, perturbing_rates, mlp_wiscon, desired_outcome, k+1)
        three = ufc.three_feature_dynamic_synthetic_data(wiscon, Xtest[t:t + 1], features_for_corr, intervals, u_f_cat_list, features, perturbing_rates, mlp_wiscon, desired_outcome, k+1)
        one.to_csv(one_path + '' + str(t) + '' + '.csv', index=False)
        two.to_csv(two_path + '' + str(t) + '' + '.csv', index=False)
        three.to_csv(three_path + '' + str(t) + '' + '.csv', index=False)
        cfcount1 += one.shape[0]  #k if one.shape[0] > k else one.shape[0]
        cfcount2 += two.shape[0] #k if two.shape[0] > k else two.shape[0]
        cfcount3 += three.shape[0] #k if three.shape[0] > k else three.shape[0]
        print(k, t, one.shape[0], two.shape[0], three.shape[0])
    validity_cf_1f[k] = cfcount1 / K
    validity_cf_2f[k] = cfcount2 / K
    validity_cf_3f[k] = cfcount3 / K
print("Validity")
print(validity_cf_1f)
print(validity_cf_2f)
print(validity_cf_3f)
'''
#### Evaluation metrics start here

features = ['ClumpThickness', 'CellSize', 'CellShape', 'MarginalAdhesion', 'EpithelialSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses']
cont_features = ['ClumpThickness', 'CellSize', 'CellShape', 'MarginalAdhesion', 'EpithelialSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses']
cat_features = []
var_features = ['ClumpThickness', 'CellSize', 'CellShape', 'MarginalAdhesion', 'EpithelialSize', 'BareNuclei', 'BlandChromatin', 'NormalNucleoli', 'Mitoses']
nbr_features = 9

### Implausibility - local outlier factor - lof

plaus_count_one = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
plaus_count_two = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
plaus_count_three = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
tempone = dict()
temptwo = dict()
tempthree = dict()
start = 0
print("Plausibility")

for t in range(K):
  try:
    one = pd.read_csv(one_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(one.shape[0]):
      if c > 20:
        break
      tempone[c] = lof(Xtest[t:t + 1], one[start:c + 1], Xtrain, scaler)
    plaus_count_one = merge_dictionaries(plaus_count_one, tempone)
  try:
    two = pd.read_csv(two_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(two.shape[0]):
      if c>20:
        break
      temptwo[c] = lof(Xtest[t:t + 1], two[start:c + 1], Xtrain, scaler)
    plaus_count_two = merge_dictionaries(plaus_count_two, temptwo)
  try:
    three = pd.read_csv(three_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(three.shape[0]):
      if c>20:
        break
      tempthree[c] = lof(Xtest[t:t + 1], three[start:c + 1], Xtrain, scaler)
    plaus_count_three = merge_dictionaries(plaus_count_three, tempthree)
print("one-feature:",plaus_count_one)
print("two-feature:",plaus_count_two)
print("three-feature:",plaus_count_three)
print("one-feature plausibility:", {i:plaus_count_one[i]/K for i in plaus_count_one})
print("two-feature plausibility:", {i:plaus_count_two[i]/K for i in plaus_count_two})
print("three-feature plausibility:", {i:plaus_count_three[i]/K for i in plaus_count_three})

## DIVERSITY

divrsity_count_one = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
divrsity_count_two = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
divrsity_count_three = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
tempone = dict()
temptwo = dict()
tempthree = dict()
start = 0
print("DIVERSITY count")
for t in range(K):
  try:
    one = pd.read_csv(one_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(one.shape[0]):
      if c>20:
        break
      tempone[c] = count_diversity(one[start:c + 1], features, nbr_features, cont_features)
    divrsity_count_one = merge_dictionaries(divrsity_count_one, tempone)
  try:
    two = pd.read_csv(two_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(two.shape[0]):
      if c>20:
        break
      temptwo[c] = count_diversity(two[start:c + 1], features, nbr_features, cont_features)
    divrsity_count_two = merge_dictionaries(divrsity_count_two, temptwo)
  try:
    three = pd.read_csv(three_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(three.shape[0]):
      if c>20:
        break
      tempthree[c] = count_diversity(three[start:c + 1], features, nbr_features, cont_features)
    divrsity_count_three = merge_dictionaries(divrsity_count_three, tempthree)
print("one-feature:", divrsity_count_one)
print("two-feature:", divrsity_count_two)
print("three-feature:", divrsity_count_three)
print("one-feature diversity-count:", {i:divrsity_count_one[i]/K for i in divrsity_count_one})
print("two-feature diversity-count:", {i:divrsity_count_two[i]/K for i in divrsity_count_two})
print("three-feature diversity-count:", {i:divrsity_count_three[i]/K for i in divrsity_count_three})

#diversity for distance

print("Diversity: Distance (div_dist) cat-cont")
diversity_count_one = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
diversity_count_two = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
diversity_count_three = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
tempone = dict()
temptwo = dict()
tempthree = dict()
start = 0
for t in range(len(Xtest[:K])):
  try:
    one = pd.read_csv(one_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(one.shape[0]):
      if c>20:
        break
      tempone[c] = diversity_l2j(one[start:c + 1], cont_features, cat_features, ratio_cont=None, agg=None)
    diversity_count_one = merge_dictionaries(diversity_count_one, tempone)
  try:
    two = pd.read_csv(two_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(two.shape[0]):
      if c>20:
        break
      temptwo[c] = diversity_l2j(two[start:c + 1], cont_features, cat_features, ratio_cont=None, agg=None)
    diversity_count_two = merge_dictionaries(diversity_count_two, temptwo)
  try:
    three = pd.read_csv(three_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(three.shape[0]):
      if c>20:
        break
      tempthree[c] = diversity_l2j(three[start:c+1], cont_features, cat_features, ratio_cont=None, agg=None)
    diversity_count_three = merge_dictionaries(diversity_count_three, tempthree)
print("one-feature:",diversity_count_one)
print("two-feature:",diversity_count_two)
print("three-feature:",diversity_count_three)
print("Diversity distance one-feature:", {i:diversity_count_one[i]/K for i in diversity_count_one})
print("Diversity distance two-feature:", {i:diversity_count_two[i]/K for i in diversity_count_two})
print("Diversity distance three feature:", {i:diversity_count_three[i]/K for i in diversity_count_three})



## SPARSITY : nbr of changes per CF

# need to make it consistent as proximity
print("Dissimilarity-SPARSITY dis_count - Change of features in CFs")
sparsity_count_one = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
sparsity_count_two = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
sparsity_count_three = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
start = 0
tempone = dict()
temptwo = dict()
tempthree = dict()
for t in range(len(Xtest[:K])):
  try:
    one = pd.read_csv(one_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(one.shape[0]):
      if c>20:
        break
      tempone[c] = avg_nbr_changes_per_cf(Xtest[t:t + 1], one[start:c + 1], cont_features)
    sparsity_count_one = merge_dictionaries(sparsity_count_one, tempone)
  try:
    two = pd.read_csv(two_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(two.shape[0]):
      if c>20:
        break
      temptwo[c] = avg_nbr_changes_per_cf(Xtest[t:t + 1], two[start:c + 1], cont_features)
    sparsity_count_two = merge_dictionaries(sparsity_count_two, temptwo)
  try:
    three = pd.read_csv(three_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(three.shape[0]):
      if c>20:
        break
      tempthree[c] = avg_nbr_changes_per_cf(Xtest[t:t + 1], three[start:c + 1], cont_features)
    sparsity_count_three = merge_dictionaries(sparsity_count_three, tempthree)
print("one-feature:", sparsity_count_one)
print("two-feature:",sparsity_count_two)
print("three-feature:",sparsity_count_three)
print("one-feature sparsity-mean:", {i:sparsity_count_one[i]/K for i in sparsity_count_one})
print("two-feature sparsity-mean:", {i:sparsity_count_two[i]/K for i in sparsity_count_two})
print("three-feature sparsity-mean:", {i:sparsity_count_three[i]/K for i in sparsity_count_three})


## DISSIMILARITY (Proximity, dis_dist, its a mean distance of all K instances for 20 counterfactuals)
print("Dissimilarity: Proximity: Distance (cont-cat) dis_dist")
proximity_count_one = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
proximity_count_two = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
proximity_count_three = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
tempone = dict()
temptwo = dict()
tempthree = dict()
start = 0
for t in range(len(Xtest[:K])):
  try:
    one = pd.read_csv(one_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(one.shape[0]):
      if c>20:
        break
      tempone[c] = distance_l2j(Xtest[t:t + 1], one[start:c + 1], cont_features, cat_features, ratio_cont=None, agg=None)
    proximity_count_one = merge_dictionaries(proximity_count_one, tempone)
  try:
    two = pd.read_csv(two_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(two.shape[0]):
      if c>20:
        break
      temptwo[c] = distance_l2j(Xtest[t:t + 1], two[start:c + 1], cont_features, cat_features, ratio_cont=None, agg=None)
    proximity_count_two = merge_dictionaries(proximity_count_two, temptwo)
  try:
    three = pd.read_csv(three_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(three.shape[0]):
      if c>20:
        break
      tempthree[c] = distance_l2j(Xtest[t:t+1], three[start:c+1], cont_features, cat_features, ratio_cont=None, agg=None)
    proximity_count_three = merge_dictionaries(proximity_count_three, tempthree)
print("one-feature:", proximity_count_one)
print("two-feature:", proximity_count_two)
print("three-feature:", proximity_count_three)
print("one-feature proximity distance:", {i:proximity_count_one[i]/K for i in proximity_count_one})
print("two-feature proximity distance:", {i:proximity_count_two[i]/K for i in proximity_count_two})
print("three-feature proximity distance:", {i:proximity_count_three[i]/K for i in proximity_count_three})

### Actionability

print("Actionability - user features as variable features")
action_count_one = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
action_count_two = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
action_count_three = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
tempone = dict()
temptwo = dict()
tempthree = dict()
start = 0
for t in range(len(Xtest[:K])):
  try:
    one = pd.read_csv(one_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(one.shape[0]):
      if c>20:
        break
      tempone[c] = nbr_actionable_cf(Xtest[t:t + 1], one[start:c + 1], features, var_features)
    action_count_one = merge_dictionaries(action_count_one, tempone)
  try:
    two = pd.read_csv(two_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(two.shape[0]):
      if c>20:
        break
      temptwo[c] = nbr_actionable_cf(Xtest[t:t + 1], two[start:c + 1], features, var_features)
    action_count_two = merge_dictionaries(action_count_two, temptwo)
  try:
    three = pd.read_csv(three_path + '' + str(t) + '' + '.csv')
  except pd.io.common.EmptyDataError:
    print('File is empty')
  else:
    for c in range(three.shape[0]):
      if c>20:
        break
      tempthree[c] = nbr_actionable_cf(Xtest[t:t+1], three[start:c+1], features, var_features)
    action_count_three = merge_dictionaries(action_count_three, tempthree)
print("one-feature:", action_count_one)
print("two-feature:", action_count_two)
print("three-feature:", action_count_three)
print("one-feature actionability:", {i:action_count_one[i]/K for i in action_count_one})
print("two-feature actionability:", {i:action_count_two[i]/K for i in action_count_two})
print("three-feature actionability:", {i:action_count_three[i]/K for i in action_count_three})

#diff = one[:1].values != two[:1].values
#result = diff.flatten().sum()



#TODO
'''
    1-Here, use the evaluation metrics for 20-cf df (one, two,three), calling the metric-functions from goodness
    2-Calculate which approach among 1f-2f-3f performed better then store it. 
    3-Also, store the other results to show as well.
    4-Address which explainer performed better by ranking on metric wise, dataset-wise
    5-Also, categorize the similar datasets (3 catefories of 5 datasets each) then account why our approach ie
    better than others on which category of datasets.   
'''

def Euclidean_Dist_df(df1, df2, cols):
  return np.linalg.norm(df1[cols].values - df2[cols].values, axis=0)

def Euclidean_Dist_Arr(arr1, arr2):
  return np.linalg.norm(arr1 - arr2)

CFs_on_Xtest = pd.DataFrame()
Notfound_Xtest = pd.DataFrame()
track_idx_found = dict()
tstinst = pd.DataFrame()


#Xtest_df = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\datasets\\totest_data\\dice_lg_diabetes_TestInstances.csv')
#Xtest_df = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\datasets\\totest_data\\dice_mlp_adult_TestInstances.csv')

#del Xtest_df['Unnamed: 0']
#print(Xtest_df.columns)

#Xtest_df = Xtest_df.iloc[1: , :]

'''
one_path = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\mammo\\lg\\one\\'
for t in range(len(Xtest_df)):
    if model.predict(Xtest_df[t:t+1]) == 1: #wine quality 0 to 1
        #print("test_intsnace_running:", Xtest_df[t:t+1].values)
        intervals = ufc.user_feedback_processing(Xtest_df[t:t+1], user_feature_list, feature_flags, threshold_values, order_of_asymmetric)
        print(intervals)
        intervals = ufc.make_interval_faithful(intervals, 'mammo')
        print(intervals)
        two = ufc.one_feature_synthetic_data_testing(Xtest_df[t:t+1], u_f_cat_list, intervals, protected_features, one_path, perturbing_rates, model)
        #two = ufc.two_feature_synthetic_data(df, Xtest_df[t:t + 1], features_for_corr, u_f_cat_list, intervals, features, two_path, perturbing_rates, model)
        #two = ufc.three_feature_dynamic_synthetic_data(df, Xtest_df[t:t + 1], features_for_corr, intervals, u_f_cat_list, features_comp, perturbing_rates, two_path, model)
        if len(two) != 0:
            CFs_on_Xtest = pd.concat([CFs_on_Xtest, two], ignore_index=True, axis=0)
            tstinst = pd.concat([tstinst, Xtest_df[t:t+1]], ignore_index=True, axis=0)
            track_idx_found[t] = Euclidean_Dist_Arr(two.values, Xtest_df[t:t+1].values)
            #print(Euclidean_Dist_Arr(one.values, Xtest_df[t:t + 1].values))
        #corr_dic, feature_cor_list = ufc.get_highly_correlated(df, features_for_corr)
        #two = ufc.two_feature_synthetic_data(df, Xtest_df[t:t + 1], features_for_corr, u_f_cat_list, intervals,
        #                                     ufcbank.features, path, ufcbank.perturbing_rates, model)
        #if len(two) != 0:
        #    CFs_on_Xtest = pd.concat([CFs_on_Xtest, two], ignore_index=True, axis=0)
        #    track_idx_found[t] = Euclidean_Dist_df(two, Xtest_df[t:t+1], ufcbank.features)
        #thr = ufc.three_feature_dynamic_synthetic_data(df, Xtest_df[t:t + 1], features_for_corr, intervals,
        #                                               u_f_cat_list, ufcbank.features, ufcbank.perturbing_rates, path, model)
        #if len(thr) != 0:
        #    CFs_on_Xtest = pd.concat([CFs_on_Xtest, thr], ignore_index=True, axis=0)
        #    track_idx_found[t] = Euclidean_Dist_df(thr, Xtest_df[t:t + 1], ufcbank.features)
ff = 'CFs_found'
fff = 'ED'
ft = 'test_instances'
CFs_on_Xtest.to_csv(one_path+''+ff+''+'.csv')
idx = pd.DataFrame.from_dict(track_idx_found, orient = 'index')
idx.to_csv(one_path+''+fff+''+'.csv')
tstinst.to_csv(one_path + '' + ft + '' + '.csv')

'''

#Outlier justification

#need to make a onepreddf in a way that the counterfactual should be added one by into the onepreddf
#and verfying at the same time from the md snd iso algothms
'''
model = ufc.train_Outliers_isolation_model(X_outlier)
outlier_count = 0
list_ids_outlier = []
for c, i in enumerate(CFs_on_Xtest):
    p = ufc.get_Outlier_isolation_prediction(model, c) #iterate here on the cf instances
    if p==1:
        outlier_count +=1
        list_ids_outlier.append(i)
print("Outliers from total CFs with IsolationForest", outlier_count, len(CFs_on_Xtest), list_ids_outlier)
'''
#mahalnobis distance outliers
#test_outliers_df = pd.concat([Xtest_df, CFs_on_Xtest], ignore_index=True, axis=0) #make it more accurate
#list_of_outliers = ufc.MD_removeOutliers(test_outliers_df) #this dataset should be the concat of the CFS and actual test instances
#print("OUTLIER INSTANCES WITH MD:", list_of_outliers)


#outlier justification

#need to make a onepreddf in a way that the counterfactual should be added one by into the onepreddf
#and verfying at the same time from the md snd iso algothms
'''
model = ufc.train_Outliers_isolation_model(X_outlier)
outlier_count = 0
list_ids_outlier = []
for c, i in enumerate(CFs_on_Xtest):
    p = ufc.get_Outlier_isolation_prediction(model, c) #iterate here on the cf instances
    if p==1:
        outlier_count +=1
        list_ids_outlier.append(i)
print("Outliers from total CFs", outlier_count, len(CFs_on_Xtest), list_ids_outlier)
'''



'''

model = ufc.train_Outliers_isolation_model(X_outlier)
outlier_count = 0
list_ids_outlier = []
for c, i in enumerate(CFs_on_Xtest):
    p = ufc.get_Outlier_isolation_prediction(model, c) #iterate here on the cf instances
    if p==1:
        outlier_count +=1
        list_ids_outlier.append(i)
print("Outliers from total CFs", outlier_count, len(CFs_on_Xtest), list_ids_outlier)

#mahalnobis distance outliers
test_outliers_df = pd.concat([Xtest_df, CFs_on_Xtest], ignore_index=True, axis=0) #make it more accurate
list_of_outliers = ufc.MD_removeOutliers(test_outliers_df) #this dataset should be the concat of the CFS and actual test instances
print("OUTLIER INSTANCES WITH MD:", list_of_outliers)



# generating natural language explanations

features, new_values, actual_values = ufc.features_to_explain(Xtest_df[:1], CFs_on_Xtest[:1])
#print("test", Xtest_df[:1])
#print("cf", CFs_on_Xtest[:1])
outcome_var = "The severity"
actual_class = 'YES'
desired_class = 'NO'
ufc.generate_reason_explanation(outcome_var, desired_class, actual_class, features)
ufc.generate_suggestion_explanation(outcome_var, desired_class, actual_class, features, new_values, actual_values)
'''

