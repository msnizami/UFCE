import numpy as np

from scipy.spatial.distance import cdist, pdist
from scipy.spatial.distance import _validate_vector
from scipy.stats import median_absolute_deviation

from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor

'''
    These utility functions are defined to evaluate different aspects of counterfactuals.
    The multiple functions are customized as per our needs and taken from a source Riccardo Guidotti latest survey conducted 
    for multiple counterfactuals methods. The source is available at this link https://github.com/riccotti/Scamander'''

# def nbr_valid_cf(cf_list, b, y_val, y_desidered=None):
#     y_cf = b.predict(cf_list)
#     idx = y_cf != y_val if y_desidered is None else y_cf == y_desidered
#     val = np.sum(idx)
#     return val

# def perc_valid_cf(cf_list, b, y_val, k=None, y_desidered=None):
#     n_val = nbr_valid_cf(cf_list, b, y_val, y_desidered)
#     k = len(cf_list) if k is None else k
#     res = n_val / k
#     return res

#self-defined
def nbr_actionable_cf(x, cf_list, features, variable_features):
    nbr_actionable = 0
    for i, cf in cf_list.iterrows():
        for j in features:
            if cf[j] != x[j].values and j in variable_features:
                nbr_actionable += 1
    return nbr_actionable/cf_list.shape[0]

def number_of_diff(df1, df2):
    differences = 0
    for i in range(len(df1)):
        if not df1.loc[i, :].equals(df2.loc[i, :]):
            differences += 1
    return differences

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

# def diversity_mh(cf_list, continuous_features, categorical_features, X, ratio_cont=None, agg=None):
#     nbr_features = cf_list.shape[1]
#     dist_cont = continuous_diversity(cf_list, continuous_features, metric='mad', X=X, agg=agg)
#     dist_cate = categorical_diversity(cf_list, categorical_features, metric='hamming', agg=agg)
#     if ratio_cont is None:
#         ratio_continuous = len(continuous_features) / nbr_features
#         ratio_categorical = len(categorical_features) / nbr_features
#     else:
#         ratio_continuous = ratio_cont
#         ratio_categorical = 1.0 - ratio_cont
#     dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
#     return dist


# def count_diversity(cf_list, features, nbr_features, continuous_features):
#     nbr_cf = cf_list.shape[0]
#     nbr_changes = 0
#     for i in range(nbr_cf):
#         for j in range(i + 1, nbr_cf):
#             for k in features:
#                 if cf_list[i][k] != cf_list[j][k]:
#                     nbr_changes += 1 if j in continuous_features else 0.5
#     return nbr_changes / (nbr_cf * nbr_cf * nbr_features)

#new updated methods
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


def avg_nbr_changes_per_cf(x, cf_list, continuous_features):
    return np.mean(nbr_changes_per_cf(x, cf_list, continuous_features))
#end here new methods


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


def plausibility(x, bb, cf_list, X_test, y_pred, continuous_features_all,
                 categorical_features_all, X_train, ratio_cont):
    sum_dist = 0.0
    for cf in cf_list:
        y_cf_val = bb.predict(cf.reshape(1, -1))[0]
        X_test_y = X_test[y_cf_val == y_pred]
        # neigh_dist = exp.cdist(x.reshape(1, -1), X_test_y)
        neigh_dist = distance_mh(x.reshape(1, -1), X_test_y, continuous_features_all,
                                 categorical_features_all, X_train, ratio_cont)
        idx_neigh = np.argsort(neigh_dist)[0]
        # closest_idx = closest_idx = idx_neigh[0]
        # closest = X_test_y[closest_idx]
        closest = X_test_y[idx_neigh]
        d = distance_mh(cf, closest.reshape(1, -1), continuous_features_all,
                        categorical_features_all, X_train, ratio_cont)
        sum_dist += d
    return sum_dist