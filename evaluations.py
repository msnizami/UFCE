from sklearn.preprocessing import StandardScaler
import numpy as np
from ufce import UFCE
ufc = UFCE()

# # joint-proximty
def Joint_proximity(oneF_cfdf, twoF_cfdf, threeF_cfdf, dice_cfs, ar_cfs, X_test, numf, catf):
    """
    :param oneF_cfdf:
    :param twoF_cfdf:
    :param threeF_cfdf:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param numf:
    :param catf:
    :return:
    """
    one_e2j = []
    for x in range(len(oneF_cfdf)):
        one_e2j.append(ufc.distance_e2j(X_test[x:x+1], oneF_cfdf[x:x+1], numf, catf, ratio_cont=None, agg=None))
    two_e2j = []
    for x in range(len(twoF_cfdf)):
        two_e2j.append(ufc.distance_e2j(X_test[x:x+1], twoF_cfdf[x:x+1], numf, catf, ratio_cont=None, agg=None))
    three_e2j = []
    for x in range(len(threeF_cfdf)):
        three_e2j.append(ufc.distance_e2j(X_test[x:x+1], threeF_cfdf[x:x+1], numf, catf, ratio_cont=None, agg=None))
    dice_e2j = []
    for x in range(len(dice_cfs)):
        dice_e2j.append(ufc.distance_e2j(X_test[x:x+1], dice_cfs[x:x+1], numf, catf, ratio_cont=None, agg=None))
    ar_e2j = []
    for x in range(len(ar_cfs)):
        ar_e2j.append(ufc.distance_e2j(X_test[x:x+1], ar_cfs[x:x+1], numf, catf, ratio_cont=None, agg=None))
    one = np.mean(one_e2j)
    two = np.mean(two_e2j)
    three = np.mean(three_e2j)
    dice = np.mean(dice_e2j)
    ar = np.mean(ar_e2j)
    means = [one, two, three, dice, ar]
    #mmeans = [i/max(means) for i in means]
    dice_std = np.std(dice_e2j) / np.sqrt(np.size(dice_e2j))
    ar_std = np.std(ar_e2j) / np.sqrt(np.size(ar_e2j))
    one_std = np.std(one_e2j) / np.sqrt(np.size(one_e2j))
    two_std = np.std(two_e2j) / np.sqrt(np.size(two_e2j))
    three_std = np.std(two_e2j) / np.sqrt(np.size(two_e2j))
    stds = [one_std, two_std, three_std, dice_std, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Catproximity(oneF_cfdf, testout1, twoF_cfdf, testout2, threeF_cfdf, testout3, dice_cfs, ar_cfs, X_test, catf):
    """
    :param oneF_cfdf:
    :param testout1:
    :param twoF_cfdf:
    :param testout2:
    :param threeF_cfdf:
    :param testout3:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param catf:
    :return:
    """
    one_e2j = []
    for x in range(len(testout1)):
        one_e2j.append(ufc.categorical_distance(testout1[x:x+1], oneF_cfdf[x:x+1], catf, metric='jaccard', agg=None))
    two_e2j = []
    for x in range(len(testout2)):
        two_e2j.append(ufc.categorical_distance(testout2[x:x+1], twoF_cfdf[x:x+1], catf, metric='jaccard', agg=None))
    three_e2j = []
    for x in range(len(testout3)):
        three_e2j.append(ufc.categorical_distance(testout3[x:x+1], threeF_cfdf[x:x+1], catf, metric='jaccard', agg=None))
    dice_e2j = []
    for x in range(len(dice_cfs)):
        dice_e2j.append(ufc.categorical_distance(X_test[x:x+1], dice_cfs[x:x+1], catf, metric='jaccard', agg=None))
    ar_e2j = []
    for x in range(len(ar_cfs)):
        ar_e2j.append(ufc.categorical_distance(X_test[x:x+1], ar_cfs[x:x+1], catf, metric='jaccard', agg=None))
    one = np.mean(one_e2j)
    two = np.mean(two_e2j)
    three = np.mean(three_e2j)
    dice = np.mean(dice_e2j)
    ar = np.mean(ar_e2j)
    means = [one, two, three, dice, ar]
    #mmeans = [i/max(means) for i in means]
    dice_std = np.std(dice_e2j) / np.sqrt(np.size(dice_e2j))
    ar_std = np.std(ar_e2j) / np.sqrt(np.size(ar_e2j))
    one_std = np.std(one_e2j) / np.sqrt(np.size(one_e2j))
    two_std = np.std(two_e2j) / np.sqrt(np.size(two_e2j))
    three_std = np.std(three_e2j) / np.sqrt(np.size(three_e2j))
    stds = [one_std, two_std, three_std, dice_std, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Contproximity(oneF_cfdf, testout1,  twoF_cfdf, testout2, threeF_cfdf, testout3, dice_cfs, ar_cfs, X_test, numf):
    """
    :param oneF_cfdf:
    :param testout1:
    :param twoF_cfdf:
    :param testout2:
    :param threeF_cfdf:
    :param testout3:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param numf:
    :return:
    """
    one_e2j = []
    for x in range(len(testout1)):
        one_e2j.append(ufc.continuous_distance(testout1[x:x+1], oneF_cfdf[x:x+1], numf, metric='euclidean', agg=None))
    two_e2j = []
    for x in range(len(testout2)):
        two_e2j.append(ufc.continuous_distance(testout2[x:x+1], twoF_cfdf[x:x+1], numf, metric='euclidean', agg=None))
    three_e2j = []
    for x in range(len(testout3)):
        three_e2j.append(ufc.continuous_distance(testout3[x:x+1], threeF_cfdf[x:x+1], numf, metric='euclidean', agg=None))
    dice_e2j = []
    for x in range(len(dice_cfs)):
        dice_e2j.append(ufc.continuous_distance(X_test[x:x+1], dice_cfs[x:x+1], numf, metric='euclidean', agg=None))
    ar_e2j = []
    for x in range(len(ar_cfs)):
        ar_e2j.append(ufc.continuous_distance(X_test[x:x+1], ar_cfs[x:x+1], numf, metric='euclidean', agg=None))
    one = np.mean(one_e2j)
    two = np.mean(two_e2j)
    three = np.mean(three_e2j)
    dice = np.mean(dice_e2j)
    ar = np.mean(ar_e2j)
    means = [one, two, three, dice, ar]
    #mmeans = [i / max(means) for i in means]
    dice_std = np.std(dice_e2j) / np.sqrt(np.size(dice_e2j))
    ar_std = np.std(ar_e2j) / np.sqrt(np.size(ar_e2j))
    one_std = np.std(one_e2j) / np.sqrt(np.size(one_e2j))
    two_std = np.std(two_e2j) / np.sqrt(np.size(two_e2j))
    three_std = np.std(three_e2j) / np.sqrt(np.size(three_e2j))
    stds = [one_std, two_std, three_std, dice_std, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Sparsity(oneF_cfdf, testout1, twoF_cfdf, testout2, threeF_cfdf, testout3, dice_cfs, ar_cfs, X_test, numf):
    """
    :param oneF_cfdf:
    :param testout1:
    :param twoF_cfdf:
    :param testout2:
    :param threeF_cfdf:
    :param testout3:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param numf:
    :return:
    """

    one_sparsity_d, one_val = ufc.sparsity_count(oneF_cfdf, len(testout1), testout1, numf)
    two_sparsity_d, two_val = ufc.sparsity_count(twoF_cfdf, len(testout2), testout2,  numf)
    three_sparsity_d, three_val = ufc.sparsity_count(threeF_cfdf, len(testout3), testout3,  numf)
    dice_sparsity_d, dice_val = ufc.sparsity_count(dice_cfs, len(dice_cfs), X_test,  numf)
    ar_sparsity_d, ar_val = ufc.sparsity_count(ar_cfs, len(ar_cfs), X_test, numf)
    dice = np.array(list(dice_sparsity_d.values())).mean()
    ar = np.array(list(ar_sparsity_d.values())).mean()
    one = np.array(list(one_sparsity_d.values())).mean()
    two = np.array(list(two_sparsity_d.values())).mean()
    three = np.array(list(three_sparsity_d.values())).mean()
    means = [one, two, three, dice, ar]
    #mmeans = [i / max(means) for i in means]
    dice_std = np.array(list(dice_sparsity_d.values())).std()
    ar_std = np.array(list(ar_sparsity_d.values())).std()
    one_std = np.array(list(one_sparsity_d.values())).std()
    two_std = np.array(list(two_sparsity_d.values())).std()
    three_std = np.array(list(three_sparsity_d.values())).std()
    stds = [one_std, two_std, three_std, dice_std, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Actionability(oneF_cfdf, testout1, twoF_cfdf, testout2, threeF_cfdf, testout3, dice_cfs, ar_cfs, X_test, features, f2change):
    """
    :param oneF_cfdf:
    :param testout1:
    :param twoF_cfdf:
    :param testout2:
    :param threeF_cfdf:
    :param testout3:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param features:
    :param f2change:
    :return:
    """
    one_actionability_d, one_val = ufc.actionability(oneF_cfdf, len(testout3), testout1, features, f2change)
    two_actionability_d, two_val = ufc.actionability(twoF_cfdf, len(testout2), testout2, features, f2change)
    three_actionability_d, three_val = ufc.actionability(threeF_cfdf, len(testout3), testout3, features, f2change)
    dice_actionability_d, dice_val = ufc.actionability(dice_cfs, len(dice_cfs), X_test, features, f2change)
    ar_actionability_d, ar_val = ufc.actionability(ar_cfs, len(ar_cfs), X_test, features, f2change)
    dice = np.array(list(dice_actionability_d.values())).mean()
    ar = np.array(list(ar_actionability_d.values())).mean()
    one = np.array(list(one_actionability_d.values())).mean()
    two = np.array(list(two_actionability_d.values())).mean()
    three = np.array(list(three_actionability_d.values())).mean()
    means = [one, two, three, dice, ar] # divider provide equal impact
    #mmeans = [i / max(means) for i in means]
    dice_std = np.array(list(dice_actionability_d.values())).std()
    ar_std = np.array(list(ar_actionability_d.values())).std()
    one_std = np.array(list(one_actionability_d.values())).std()
    two_std = np.array(list(two_actionability_d.values())).std()
    three_std = np.array(list(three_actionability_d.values())).std()
    stds = [one_std, two_std, three_std, dice_std, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Plausibility(oneF_cfdf, twoF_cfdf, threeF_cfdf, dice_cfs, ar_cfs, X_test, X_train):
    """
    :param oneF_cfdf:
    :param twoF_cfdf:
    :param threeF_cfdf:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param X_train:
    :return:
    """
    one_plausibility_d, one_val = ufc.implausibility(oneF_cfdf, X_test, X_train[:], len(oneF_cfdf))
    two_plausibility_d, two_val = ufc.implausibility(twoF_cfdf, X_test, X_train[:], len(twoF_cfdf))
    three_plausibility_d, three_val = ufc.implausibility(threeF_cfdf, X_test, X_train[:], len(threeF_cfdf))
    dice_plausibility_d, dice_val = ufc.implausibility(dice_cfs, X_test, X_train[:], len(dice_cfs))
    ar_plausibility_d, ar_val = ufc.implausibility(ar_cfs, X_test, X_train[:], len(ar_cfs))
    dice = np.array(list(dice_plausibility_d.values())).mean()
    # ar = np.array(list(ar_plausibility_d.values())).mean()
    # one = np.array(list(one_plausibility_d.values())).mean()
    # two = np.array(list(two_plausibility_d.values())).mean()
    # three = np.array(list(three_plausibility_d.values())).mean()
    means = [one_val, two_val, three_val, dice_val, ar_val]
    #mmeans = [i / max(means) for i in means]
    dice_std = np.array(list(dice_plausibility_d.values())).std()
    ar_std = np.array(list(ar_plausibility_d.values())).std()
    one_std = np.array(list(one_plausibility_d.values())).std()
    two_std = np.array(list(two_plausibility_d.values())).std()
    three_std = np.array(list(three_plausibility_d.values())).std()
    stds = [one_std, two_std, three_std, dice_std, ar_std]
    #mstds = [j / max(stds) for j in stds]
    return means, stds

def Feasibility(oneF_cfdf, twoF_cfdf, threeF_cfdf, dice_cfs, ar_cfs, X_test, X_train, features, f2change, bb, desired_outcome, outcome_label):
    """
    :param oneF_cfdf:
    :param twoF_cfdf:
    :param threeF_cfdf:
    :param dice_cfs:
    :param ar_cfs:
    :param X_test:
    :param X_train:
    :param features:
    :param f2change:
    :param bb: black box model
    :param desired_outcome:
    :param outcome_label:
    :return:
    """
    dice = ufc.feasibility(X_test, dice_cfs, X_train, features, f2change, bb, desired_outcome, label=outcome_label, dice=False)
    ar = ufc.feasibility(X_test, ar_cfs, X_train, features, f2change, bb, desired_outcome, label=outcome_label, dice=False)
    one = ufc.feasibility(X_test, oneF_cfdf, X_train, features, f2change, bb, desired_outcome, label=outcome_label, dice=False)
    two = ufc.feasibility(X_test, twoF_cfdf, X_train, features, f2change, bb, desired_outcome, label=outcome_label, dice=False)
    three = ufc.feasibility(X_test, threeF_cfdf, X_train, features, f2change, bb, desired_outcome, label=outcome_label, dice=False)
    means = [one, two, three, dice, ar]
    stds = [0,0,0,0,0]
    return means, stds
