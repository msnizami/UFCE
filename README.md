# UFCE
user feedback-based counterfactual explanations

This ReadMe file provides an overview of the main steps and model code snippets. The complete source code and all the utilities are availble in the `src` folder. 
The most of utilities/helper functions are exploited from `ufce.py` file (`ufce.py` file contains multiple methods of previous approach too).

Running the experiment:
The file `CF-prototype-based-approach-src.py` can be run in PyCHarm IDE or the file `CF-prototype-based-approach-src.ipynb` can be run in Jupyetr Notebook.
The datasets can be changed and the parameters need to change accordingly (e.g., user-feedback constraints).

This example provides demo on Bank Loan dataset in the Data/ repository. A step by step demo for bank loan data is provided with complete details in the file `CF-prototype-based-approach-src.ipynb`.

Import libraries.
```python
import time
import gower
import random
import pandas as pd
import numpy as np
from goodness import *
import seaborn as sns
from numpy import meshgrid
from numpy import arange
from numpy import hstack
from scipy import stats
%matplotlib inline 
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.metrics import balanced_accuracy_score
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_absolute_deviation
from sklearn.neighbors import LocalOutlierFactor
from pandas.errors import EmptyDataError
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
plt.style.use("seaborn-whitegrid")
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
```
Load data
```
path = r'C:\Users\laboratorio\Documents\Suffian PhD Work\codes\datasets\Bank_Loan.csv'
bankloan = pd.read_csv(path)
del bankloan['Unnamed: 0']
del bankloan['age']
del bankloan['Experience']
```
Collecting Label-1 data points (desired space)
```
data_lab1 = pd.DataFrame()
data_lab1 = bankloan[bankloan["Personal Loan"] == 1]
data_lab0 = bankloan[bankloan["Personal Loan"] == 0]
data_lab1 = data_lab1.drop(['Personal Loan'], axis=1)
len(data_lab1), len(data_lab0)
```
Importing functionality of UFCE
```
import ufce
from goodness import *
from ufce import UFCE
ufc = UFCE()
```
Finding Mutual Information
```
Y = bankloan.loc[ : , bankloan.columns == 'Personal Loan'] 
X = bankloan.loc[ : , bankloan.columns != 'Personal Loan']
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()
discrete_features = X.dtypes == int
# mi_scores = ufc.make_mi_scores(X, Y, discrete_features) # this snippet of code provides individual importance scores
# mi_scores  # show a few features with their MI scores
# get top mutual information sharing pair of feature
features = ['Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
       'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
cat_f = ['SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
F = ufc.get_top_MI_features(X, features)
F[:5]
```
10-fold Cross-validation
```
from sklearn.model_selection import cross_val_score
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
scores = cross_val_score(lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print('Cross Validation accuracy scores: %s' % scores)
print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
```
Train-test splits with a random state that provides the best distribution fit of data
```
from scipy import stats
from sklearn.model_selection import train_test_split
n_features = X.shape[1]
n_tries = 5
result = []
for random_state in range(n_tries):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=random_state)
    distances = list(map(lambda i : stats.ks_2samp(X_train.iloc[:,i],X_test.iloc[:,i]).statistic,range(n_features)))
    result.append((random_state,max(distances)))
result.sort(key = lambda x : x[1])
# from result, random state is important to train the model on the best split
idx = 0
random_state = result[idx][0]  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=random_state)
X_test.to_csv(path + 'X_test' + '.csv', index=False)
X_train.to_csv(path + 'X_train' + '.csv', index=False)
```

Training the ML models
```
lr = LogisticRegression(max_iter=1000)
model = MLPClassifier(max_iter=1000)
model.fit(X_train,y_train.values.ravel())
lr.fit(X_train,y_train.values.ravel())
print("LR R2 score",r2_score(y_test.values.ravel(),lr.predict(X_test)))
print("LR accuracy",accuracy_score(y_test.values.ravel(), lr.predict(X_test)))
```

User feedback constraints (these constraints were subjected equally for each test nstance) 
```
uf = {'Income':70, 'CCAvg':3.0, 'Family': 3, 'Education':3, 'Mortgage':100, 'CDAccount':1,'Online':1, 'SecuritiesAccount':1, 'CreditCard':1}
catf = ['CDAccount','Online', 'SecuritiesAccount', 'CreditCard']
numf = ['Income', 'CCAvg', 'Family', 'Education', 'Mortgage']
f2change = ['Income', 'CCAvg', 'Mortgage', 'CDAccount', 'Online']
desired_outcome = 1.0
k = 1 #no. of counterfactuals
```
Finding Nearest Neighbors
```
nn, idx = ufc.NNkdtree(data_lab1, X_test[:1], 100)
cfs = ufc.get_cfs_validated(nn, lr, 1)
# len(cfs)
# features to change
ufc.feat2change(X_test[:1], nn[0:1])
```
Generating counterfatuals with UFCE1
```
oneF_cfdf = pd.DataFrame()
onetest_cfdf = pd.DataFrame()
found_indexes = []
start = time.time()
for t in range(len(X_test[:1])):
    n = 0
    nn, idx = ufc.NNkdtree(data_lab1, X_test[t:t+1], 100) #increase radius size as per the dataset
    #cfs = get_cfs_validated(nn, mlp, desired_outcome) # in case if cfs are not avalable then use nn
    if nn.empty != True:
        interval = ufc.make_intervals(nn, n, uf, f2change, X_test[t:t+1], catf, numf) # also use cfs instead of nn
        cc= ufc.one_feature_binsearch(X_test[t:t+1], catf, numf, interval, lr, desired_outcome, k)
        while cc.empty == True:
            n = n+1
            interval = ufc.make_intervals(nn, n, uf, f2change, X_test[t:t+1], catf, numf) #also use cfs instead of nn
            cc = ufc.one_feature_binsearch(X_test[t:t+1], catf, numf, interval, lr, desired_outcome, k)
            if n >= 10:
                break
        if cc.empty != True:
            found_indexes.append(t)
            oneF_cfdf = pd.concat([oneF_cfdf, cc[:1]], ignore_index=True, axis=0)
end = time.time()
onetime = end - start
print('ufce1 time', onetime)
```
Generating counterfactuals with UFCE2
```
start = time.time()
n = 0
perturb_step = {} 
twoF_cfdf = pd.DataFrame()
protectedf = [] # take empty

# running th experiment for mutliple test instances (at-least 50 for comparison)
for t in range(len(X_test[:1])):
    n=0
    nn, idx = ufc.NNkdtree(data_lab1, X_test[t:t+1], 100)
    if nn.empty != True:
        intervals = ufc.make_uf_nn_interval(uf, F[:5], nn, 5, X_test[t:t+1])
        cc2, cfsexp2 = ufc.two_feature_update_corr_reg_binsearch(X, X_test[t:t+1], protectedf, F[:5], catf, numf, intervals, features, perturb_step, lr, desired_outcome, k)
        while cc2.empty == True:
            n = n+1
            intervals = ufc.make_uf_nn_interval(uf, F[:5], nn, 5, X_test[t:t+1])
            cc2, cfsexp2 = ufc.two_feature_update_corr_reg_binsearch(X, X_test[t:t+1], protectedf, F[:5], catf, numf, intervals, features, perturb_step, lr, desired_outcome, k)
            if n >= 15:
                break
        if cc2.empty == True:
            cnn = nn[n:n+1].values
            twoF_cfdf = pd.concat([twoF_cfdf, nn[n:n+1]], ignore_index=True, axis=0)
        else:
            cnn = []
        twoF_cfdf = pd.concat([twoF_cfdf, cc2[:1]], ignore_index=True, axis=0)
end = time.time()
twotime = end-start
print('ufce2 time:', twotime)
```
Generating counterfactuals with UFCE3
```
start = time.time()
perturb_step = {}
n = 0
protectedf = [] # take empty
feature_pairs = F[:5]
# running th experiment for mutliple test instances (at-least 50 for comparison)
threeF_cfdf = pd.DataFrame()
for t in range(len(X_test[:1])):
    n=0
    nn, idx = ufc.NNkdtree(data_lab1, X_test[t:t+1], 100)
    if nn.empty != True:
        intervals = ufc.make_uf_nn_interval(uf, feature_pairs, nn, 5, X_test[t:t+1]) # cfs instead nn
        cc2, cfsexp2 = ufc.three_feature_update_corr_reg_binsearch(X, X_test[t:t+1], protectedf, feature_pairs, catf, numf, intervals, features, perturb_step, lr, desired_outcome, k)
        while cc2.empty == True:
            n = n+1
            intervals = ufc.make_uf_nn_interval(uf, feature_pairs, nn, 5, X_test[t:t+1])
            cc2, cfsexp2 = ufc.three_feature_update_corr_reg_binsearch(X, X_test[t:t+1], protectedf, feature_pairs, catf, numf, intervals, features, perturb_step, lr, desired_outcome, k)
            if n >= 10:
                break
        if cc2.empty == True:
            cnn = nn[n:n+1].values
            threeF_cfdf = pd.concat([threeF_cfdf, nn[n:n+1]], ignore_index=True, axis=0)
        else:
            cnn = []
        threeF_cfdf = pd.concat([threeF_cfdf, cc2[:1]], ignore_index=True, axis=0)
end = time.time()
threetime = end-start
print('ufce3 time:', threetime)
```
Generating counterfactuals with DiCE

fro DiCE, need to install its package (pip install dice_ml)
```
import time
import dice_ml
from dice_ml.utils import helpers # helper functions
from sklearn.model_selection import train_test_split
start = time.time()
d = dice_ml.Data(dataframe=bankloan, continuous_features=numf, outcome_name= 'Personal Loan')
m = dice_ml.Model(model=lr, backend="sklearn")
exp = dice_ml.Dice(d, m, method="random")
dice_cfs = pd.DataFrame()
for x in range(2):
    e1 = exp.generate_counterfactuals(X_test[x:x+1], total_CFs=1, desired_class="opposite", features_to_vary= features)
    cf = e1.cf_examples_list[0].final_cfs_df[0:1]
    dice_cfs = pd.concat([dice_cfs, cf], ignore_index = True, axis = 0)
end = time.time()
dicetime = end-start
print('dice time:', dicetime)
```
Generating counterfactuals with AR method
for AR, we need to install its package (pip install actionable-recourse)
```
import recourse as rs
start = time.time()
A = rs.ActionSet(X)
X = bankloan.loc[ : , bankloan.columns != 'Personal Loan']
y = bankloan['Personal Loan']
# train a classifier
clf = LogisticRegression(max_iter = 500)
clf.fit(X, y)
A.set_alignment(clf)
ar_cfs = pd.DataFrame()
for x in range(2):
    fs = rs.Flipset(X_test[x:x+1].values, action_set = A, clf = clf)
    fs.populate(enumeration_type='distinct_subsets', total_items = 1) #'mutually_exclusive'
    f_list = numf #['Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    feat2change = fs.df['features']
    values_2change = fs.df['x_new']
    changed_instance = X_test[x:x+1].copy()
    d = {0:'Income', 1:'Family', 2:'CCAvg', 3:'Education', 4:'Mortgage', 5:'Securities Account', 6:'CD Account', 7:'Online', 8:'CreditCard'}
    for f, i in enumerate(feat2change):
        changed_instance[i] = values_2change[f]
    ar_cfs = pd.concat([ar_cfs, changed_instance], ignore_index = True, axis = 0)
end = time.time()
artime = end-start
print('AR time:', artime)
```
Generating counterfactual with CEM 
I used results already computed from the Google.Colab, CEM package was working fine on colab (Also, for few plots, results are ficticious for this specific example demo).

Efficacy of CF methods

Sparsity

For example demo, only one evaluation metric is displayed here, for complete evaluation metrics and results (see file CF.ipynb)
```
one_sparsity_d, one_val = ufc.sparsity_count(oneF_cfdf, len(oneF_cfdf), X_test, numf)
two_sparsity_d, two_val = ufc.sparsity_count(twoF_cfdf, len(twoF_cfdf), X_test,  numf)
three_sparsity_d, three_val = ufc.sparsity_count(threeF_cfdf, len(threeF_cfdf), X_test,  numf)
dice_sparsity_d, dice_val = ufc.sparsity_count(dice_cfs, len(dice_cfs), X_test,  numf)
ar_sparsity_d, ar_val = ufc.sparsity_count(ar_cfs, len(ar_cfs), X_test, numf)
dice = np.array(list(dice_sparsity_d.values())).mean()
ar = np.array(list(ar_sparsity_d.values())).mean()
one = np.array(list(one_sparsity_d.values())).mean()
two = np.array(list(two_sparsity_d.values())).mean()
three = np.array(list(three_sparsity_d.values())).mean()
cem = 2.4 # from colab
dice_std = np.array(list(dice_sparsity_d.values())).std()
ar_std = np.array(list(ar_sparsity_d.values())).std()
one_std = np.array(list(one_sparsity_d.values())).std()
two_std = np.array(list(two_sparsity_d.values())).std()
three_std = np.array(list(three_sparsity_d.values())).std()
cem_std = 1.1 #from colab
methods = ['DiCE','AR','UFCE1','UFCE2','UFCE3', 'CEM']
x_pos = np.arange(len(methods))
CTEs = [dice/len(features), ar/len(features), one/len(features), two/len(features), three/len(features), cem/len(features)]
error = [dice_std/len(features), ar_std/len(features), one_std/len(features), two_std/len(features), three_std/len(features), cem_std/len(features)]
# Build the plot
ufc.barplot(methods, CTEs, x_pos, error, 'Sparsity', 'lower is the better', save=False)
```
Explanation generation in natural language (using pylng, simplenlg)
```
outcome_var = "The personal loan"
actual_class = 'denied'
desired_class = 'accepted'
test = pd.DataFrame(data=[[83, 4, 2, 3, 0, 0, 0, 1, 0]], columns =['Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
       'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard'] )
cf1 = pd.DataFrame(data=[[132.34, 4, 2, 3, 103, 0, 0, 1, 0]], columns =['Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
       'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard'] )
features, new_values, actual_values = ufc.features_to_explain(test, cf1) # A specific case is used, more cases can be used by changing the parameters to test-->Xtest and cf1-->cfs_from_any_cf_method.
ufc.generate_reason_explanation(outcome_var, desired_class, actual_class, features)
ufc.generate_suggestion_explanation(outcome_var, desired_class, actual_class, features, new_values, actual_values)
```

