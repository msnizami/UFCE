import numpy as np

from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_absolute_deviation
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor



def one_feature_binsearch(test_instance, u_cat_f_list, numf, user_term_intervals, model, outcome, k):

    one_feature_dataframe = pd.DataFrame()
    cfdfout = pd.DataFrame()
    tempdf = pd.DataFrame()
    tempdfcat = pd.DataFrame()
    one_all_explor = pd.DataFrame()
    #tempdf = test_instance.copy()
    found = 0

    #print(model.predict(test_instance))

    for feature in user_term_intervals.keys():
        if feature not in u_cat_f_list:
            #delta = 0
            i = 0
            #print(feature)
            tempdf = test_instance.copy()
            one_feature_data = pd.DataFrame()
            interval_term_range = user_term_intervals[feature]
            #print(interval_term_range, feature)
            if len(interval_term_range) != 0 and interval_term_range[0] != interval_term_range[1]:
                start = interval_term_range[0]
                end = interval_term_range[1]
                #number_of_iterations = (end - start) / perturbing_rates[feature] #closed due to binsearch below
                def binarySearch(model, outcome, start, end):
                    #print(start, end)
                    if end >= start:
                        cfdf = pd.DataFrame()
                        mid = start + (end - start)/2
                        # If found at mid, then return it
                        tempdf.loc[:, feature] = mid
                        #print(tempdf.values)
                        #one_all_explor = pd.concat([one_all_explor, tempdf], ignore_index=True, axis=0)
                        pred = model.predict(tempdf)
                        #print(pred)
                        if pred == outcome:
                            cfdf = pd.concat([cfdf, tempdf], ignore_index=True, axis=0)
#                                 found = found + 1
                            return cfdf
                        # Search the right half
                        else:
                            return binarySearch(model, outcome, mid + 1, end)
                #calling
                cfs = binarySearch(model, outcome, start, end)
                cfdfout = pd.concat([cfdfout, cfs], ignore_index=True, axis=0)
        else:
            tempdfcat = test_instance.copy()
            if found == k:
                break
            else:
                tempdfcat.loc[:, feature] = 1.0 if tempdfcat.loc[:, feature].values else 1.0
                one_all_explor = pd.concat([one_all_explor, tempdfcat], ignore_index=True, axis=0)
                pred = model.predict(tempdfcat)
                #print(pred)
                if pred == outcome:
                    cfdfout = pd.concat([cfdfout, tempdfcat], ignore_index=True, axis=0)
#                         found = found + 1
    return cfdfout, test_instance

def nbr_valid_cf(cf_list, b, y_val, y_desidered=None):
    y_cf = b.predict(cf_list)
    idx = y_cf != y_val if y_desidered is None else y_cf == y_desidered
    val = np.sum(idx)
    return val


def perc_valid_cf(cf_list, b, y_val, k=None, y_desidered=None):
    n_val = nbr_valid_cf(cf_list, b, y_val, y_desidered)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res

#self-defined
def nbr_actionable_cf(x, cf_list, features, variable_features):
    nbr_actionable = 0
    for i, cf in cf_list.iterrows():
        for j in features:
            if cf[j] != x[j].values and j in variable_features:
                nbr_actionable += 1
    return nbr_actionable/cf_list.shape[0]

    '''
    nbr_actionable = 0
    nbr_features = cf_list.shape[1]
    if cf_list.shape[0] > 1:
        for i, cf in enumerate(cf_list.values):
            constraint_violated = False
            for j in range(nbr_features):
                if cf[j] != x.iloc[:,j].values:
                    constraint_violated = True
                    break
            if not constraint_violated:
                nbr_actionable += 1
    else:
        for i, cf in enumerate(cf_list):
            constraint_violated = False
            if cf_list[cf].values != x[cf].values:
                constraint_violated = True
                break
        if not constraint_violated:
            nbr_actionable += 1
    return nbr_actionable
    '''

def number_of_diff(df1, df2):
    differences = 0
    for i in range(len(df1)):
        if not df1.loc[i, :].equals(df2.loc[i, :]):
            differences += 1
    return differences

def perc_actionable_cf(x, cf_list, variable_features, k=None):
    n_val = nbr_actionable_cf(x, cf_list, variable_features)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res


def nbr_valid_actionable_cf(x, cf_list, b, y_val, variable_features, y_desidered=None):
    y_cf = b.predict(cf_list)
    idx = y_cf != y_val if y_desidered is None else y_cf == y_desidered

    nbr_valid_actionable = 0
    nbr_features = cf_list.shape[1]
    for i, cf in enumerate(cf_list):
        if not np.array(idx)[i]:
            continue
        constraint_violated = False
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                constraint_violated = True
                break
        if not constraint_violated:
            nbr_valid_actionable += 1

    return nbr_valid_actionable


def perc_valid_actionable_cf(x, cf_list, b, y_val, variable_features, k=None, y_desidered=None):
    n_val = nbr_valid_actionable_cf(x, cf_list, b, y_val, variable_features, y_desidered)
    k = len(cf_list) if k is None else k
    res = n_val / k
    return res


def nbr_violations_per_cf(x, cf_list, variable_features):
    nbr_features = cf_list.shape[1]
    nbr_violations = np.zeros(len(cf_list))
    for i, cf in enumerate(cf_list):
        for j in range(nbr_features):
            if cf[j] != x[j] and j not in variable_features:
                nbr_violations[i] += 1
    return nbr_violations


def avg_nbr_violations_per_cf(x, cf_list, variable_features):
    return np.mean(nbr_violations_per_cf(x, cf_list, variable_features))


def avg_nbr_violations(x, cf_list, variable_features):
    val = np.sum(nbr_violations_per_cf(x, cf_list, variable_features))
    nbr_cf, nbr_features = cf_list.shape
    return val / (nbr_cf * nbr_features)


def mad_cityblock(u, v, mad):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def continuous_distance(x, cf_list, continuous_features, metric='euclidean', X=None, agg=None):
    if metric == 'mad':
        mad = median_absolute_deviation(X[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)

        dist = cdist(x.reshape(1, -1)[:, continuous_features], cf_list[:, continuous_features], metric=_mad_cityblock)
    else:
        dist = cdist(x.loc[:, continuous_features], cf_list.loc[:, continuous_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=None):
    dist = cdist(x.loc[:, categorical_features], cf_list.loc[:, categorical_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def distance_l2j(x, cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(x, cf_list, continuous_features, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

def merge_dictionaries(dict1, dict2):
    merged_dictionary = {}

    for key in dict1:
        if key in dict2:
            new_value = dict1[key] + dict2[key]
        else:
            new_value = dict1[key]

        merged_dictionary[key] = new_value

    #for key in dict2:
    #    if key not in merged_dictionary:
    #        merged_dictionary[key] = dict2[key]

    return merged_dictionary

def distance_mh(x, cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(x, cf_list, continuous_features, metric='mad', X=X, agg=agg)
    dist_cate = categorical_distance(x, cf_list, categorical_features, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist

#cf feature changes and avg change
def nbr_changes_per_cf(x, cf_list, continuous_features):
    nbr_features = cf_list.shape[1]
    nbr_changes = np.zeros(len(cf_list))
    for i, cf in cf_list.iterrows():
        for j in continuous_features:
            if cf[j] != x[j].values:
                nbr_changes[i] += 1 if j in continuous_features else 0.5
    return nbr_changes


def avg_nbr_changes_per_cf(x, cf_list, continuous_features):
    return np.mean(nbr_changes_per_cf(x, cf_list, continuous_features))


def avg_nbr_changes(x, cf_list, nbr_features, continuous_features):
    val = np.sum(nbr_changes_per_cf(x, cf_list, continuous_features))
    nbr_cf, _ = cf_list.shape
    return val / (nbr_cf * nbr_features)


def continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=None):
    if metric == 'mad':
        mad = median_absolute_deviation(X[:, continuous_features], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)

        dist = pdist(cf_list[:, continuous_features], metric=_mad_cityblock)
    else:
        dist = pdist(cf_list.loc[:, continuous_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=None):
    dist = pdist(cf_list.loc[:, categorical_features], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def diversity_l2j(cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, continuous_features, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def diversity_mh(cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, continuous_features, metric='mad', X=X, agg=agg)
    dist_cate = categorical_diversity(cf_list, categorical_features, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(continuous_features) / nbr_features
        ratio_categorical = len(categorical_features) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


'''def count_diversity(cf_list, features, nbr_features, continuous_features):
    nbr_cf = cf_list.shape[0]
    nbr_changes = 0
    for i in range(nbr_cf):
        for j in range(i + 1, nbr_cf):
            for k in features:
                if cf_list[i][k] != cf_list[j][k]:
                    nbr_changes += 1 if j in continuous_features else 0.5
    return nbr_changes / (nbr_cf * nbr_cf * nbr_features)'''

#updates
def count_diversity(cf_list, features, nbr_features, continuous_features):
    nbr_cf = cf_list.shape[0]
    nbr_changes = 0
    for i in range(nbr_cf):
        for j in range(i + 1, nbr_cf):
            for k in features:
                if cf_list[i:i+1][k].values != cf_list[j:j+1][k].values:
                    nbr_changes += 1 if j in continuous_features else 0.5
                #print("value change is ", nbr_changes)
    return nbr_changes / (nbr_cf * nbr_cf * nbr_features) if nbr_changes != 0 else 0.0
def count_diversity_all(cf_list, nbr_features, continuous_features):
    return count_diversity(cf_list, range(cf_list.shape[1]), nbr_features, continuous_features)

#cf feature changes and avg change
def nbr_changes_per_cf(x, cf_list, continuous_features):
    nbr_features = cf_list.shape[1]
    nbr_changes = np.zeros(len(cf_list))
    for i, cf in cf_list.iterrows():

        for j in continuous_features:
            if cf[j] != x[j].values:
                nbr_changes[i] += 1 if j in continuous_features else 0.5
    return nbr_changes


def avg_nbr_changes_per_cf(x, cf_list, continuous_features):
    return np.mean(nbr_changes_per_cf(x, cf_list, continuous_features))


def lof(x, cf_list, X, scaler):
    X_train = np.vstack([x.values.reshape(1, -1), X])

    nX_train = scaler.transform(X_train) #instead of X_train
    ncf_list = scaler.transform(cf_list)

    clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
    clf.fit(nX_train)
    #print("here",nX_train.shape, ncf_list.shape)
    lof_values = clf.predict(ncf_list)
    print("lof here:", lof_values)
    return np.mean(np.abs(lof_values))


