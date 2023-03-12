# pip install dice_ml
# pip install actionable-recourse

import time
import pandas as pd
import dice_ml
import recourse as rs
from dice_ml.utils import helpers # helper functions
from sklearn.model_selection import train_test_split

from ufce import UFCE
ufc = UFCE()

def dice_cfexp(df, X_test, numf, f2change, outcome_label, no_cf, bb):
    """
    :param df: dataset
    :param X_test: test set
    :param numf: numerical features
    :param f2change: features to change
    :param outcome_label: class label
    :param no_cf: required number of counterfactuals
    :param bb: blackbox model
    :return dice_cfs: dice counterfactuals
    """
    start = time.time()
    d = dice_ml.Data(dataframe=df, continuous_features=numf, outcome_name= outcome_label)
    m = dice_ml.Model(model=bb, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    dice_cfs = pd.DataFrame()
    for x in range(len(X_test)):
        e1 = exp.generate_counterfactuals(X_test[x:x+1], total_CFs=no_cf, desired_class="opposite", features_to_vary= f2change)
        #e1.cf_examples_list[0].final_cfs_df[0:1].to_csv(path + '' + str(x) + '' + '.csv', index=False)
        cf = e1.cf_examples_list[0].final_cfs_df[0:no_cf]
        dice_cfs = pd.concat([dice_cfs, cf], ignore_index = True, axis = 0)
    #dice_cfs.to_csv(r'C:\Users\~plots\ufc-dice\DiCE_mlp'+'.csv')
    end = time.time()
    dicetime = end-start
    if len(X_test) != 0:
        dicetime = dicetime/len(X_test)
    print('\t\t dice time:', dicetime)
    return dice_cfs, dicetime

def ar_cfexp(X, numf, bb, X_test):
    """
    :param X: X data
    :param numf: numerical features
    :param bb: blackboc=x model
    :param X_test: test set
    :return ar_cfs: ar counterfactuals
    """
    start = time.time()
    A = rs.ActionSet(X)
    from IPython.core.display import display, HTML
    clf = bb
    A.set_alignment(clf)
    ar_cfs = pd.DataFrame()
    for x in range(len(X_test)):
        fs = rs.Flipset(X_test[x:x+1].values, action_set = A, clf = clf)
        fs.populate(enumeration_type='distinct_subsets', total_items = 1) #'mutually_exclusive'
        f_list = numf #['Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
        #features that need to be changed and could flip the outcome bt fs
        feat2change = fs.df['features']
        values_2change = fs.df['x_new']
        changed_instance = X_test[x:x+1].copy()

        for f, i in enumerate(feat2change):
            changed_instance[i] = values_2change[f]
        ar_cfs = pd.concat([ar_cfs, changed_instance], ignore_index = True, axis = 0)

    #ar_cfs.to_csv(r'C:\Users\~\plots\ufc-dice\AR_mlp'+'.csv')
    end = time.time()
    artime = end-start
    if len(X_test) != 0:
        artime = artime/len(X_test)
    print('\t\t AR time:', artime)
    return ar_cfs, artime

def sfexp(X, data_lab1, X_test, uf, f2change, numf, catf, bb, desired_outcome, k):
    """
    :param X:
    :param data_lab1:
    :param X_test:
    :param uf:
    :param f2change:
    :param numf:
    :param catf:
    :param bb:
    :param desired_outcome:
    :param k:
    :return:
    """
    oneF_cfdf = pd.DataFrame()
    testout = pd.DataFrame()
    found_indexes = []
    intervald = dict()
    start = time.time()
    for t in range(len(X_test)):
        n = 0
        nn, idx = ufc.NNkdtree(data_lab1, X_test[t:t+1], 100) #increase radius size as per the dataset
        if nn.empty != True:
            interval = ufc.make_intervals(nn, uf, f2change, X_test[t:t+1])
            cc = ufc.Single_F(X_test[t:t+1], catf, interval, bb, desired_outcome)
            while cc.empty == True:
                n = n+1
                interval = ufc.make_intervals(nn, uf, f2change, X_test[t:t+1])
                cc = ufc.Single_F(X_test[t:t+1], catf, interval, bb, desired_outcome)
                if n >= 10:
                    break
            if cc.empty != True:
                found_indexes.append(t)
                intervald[t] = interval
                oneF_cfdf = pd.concat([oneF_cfdf, cc[:1]], ignore_index=True, axis=0)
                testout = pd.concat([testout, X_test[t:t+1]], ignore_index=True, axis=0)
        #onetest_cfdf = pd.concat([onetest_cfdf, test[:1]], ignore_index=True, axis=0)
        #oneF_cfdf.to_csv(r'C:\Users\~\plots\ufc-dice\OneF_mlp'+'.csv')
    end = time.time()
    onetime = end - start
    print('\t\t ufce1 time', onetime)
    if len(X_test) != 0:
        onetime = onetime/len(X_test)
    return oneF_cfdf, onetime, found_indexes, intervald, testout

def dfexp(X, data_lab1, X_test, uf, F, numf, catf, features, protectf, bb, desired_outcome, k):
    start = time.time()
    desired_outcome = desired_outcome
    k = k
    foundidx = []
    intervald = dict()
    perturb_step = {}
    twoF_cfdf = pd.DataFrame()
    testout = pd.DataFrame()
    protectedf = protectf

    # running th experiment for mutliple test instances (at-least 50 for comparison)
    for t in range(len(X_test)):
        n=0
        nn, idx = ufc.NNkdtree(data_lab1, X_test[t:t+1], 100)

        if nn.empty != True:
            intervals = ufc.make_uf_nn_interval(nn, uf, F[:5], X_test[t:t+1])
            cc2, cfsexp2 = ufc.Double_F(X, X_test[t:t+1], protectedf, F[:5], catf, numf, intervals, features, bb, desired_outcome)
            if cc2.empty != True:
                foundidx.append(t)
                intervald[t] = intervals
                testout = pd.concat([testout, X_test[t:t + 1]], ignore_index=True, axis=0)
            while cc2.empty == True:
                n = n+1
                intervals = ufc.make_uf_nn_interval(nn, uf, F[:5], X_test[t:t+1])
                cc2, cfsexp2 = ufc.Double_F(X, X_test[t:t+1], protectedf, F[:5], catf, numf, intervals, features, bb, desired_outcome)
                if n >= 10:
                    break
            if cc2.empty == True:
                twoF_cfdf = pd.concat([twoF_cfdf, nn[n:n+1]], ignore_index=True, axis=0)
            twoF_cfdf = pd.concat([twoF_cfdf, cc2[:1]], ignore_index=True, axis=0)
            #twoF_cfdf.to_csv(r'C:\Users\~\plots\ufc-dice\twoF'+'.csv')
    end = time.time()
    twotime = end-start
    if len(X_test) != 0:
        twotime = twotime/len(X_test)
    print('\t\t ufce2 time:', twotime)
    return twoF_cfdf, twotime, foundidx, intervald, testout

def tfexp(X, data_lab1, X_test, uf, F, numf, catf, features, protectdf, bb, desired_outcome, k):
    """
    :param X:
    :param data_lab1:
    :param X_test:
    :param uf:
    :param F:
    :param numf:
    :param catf:
    :param features:
    :param protectdf:
    :param bb:
    :param desired_outcome:
    :param k:
    :return:
    """
    start = time.time()
    perturb_step = {}
    foundidx = []
    intervald = dict()
    threeF_cfdf = pd.DataFrame()
    testout = pd.DataFrame()
    for t in range(len(X_test)):
        n=0
        nn, idx = ufc.NNkdtree(data_lab1, X_test[t:t+1], 100)
        if nn.empty != True:
            intervals = ufc.make_uf_nn_interval(nn, uf, F[:5], X_test[t:t+1]) 
            cc2, cfsexp2 = ufc.Triple_F(X, X_test[t:t+1], protectdf, F, catf, numf, intervals, features, bb, desired_outcome)
            if cc2.empty != True:
                foundidx.append(t)
                intervald[t] = intervals
                testout = pd.concat([testout, X_test[t:t + 1]], ignore_index=True, axis=0)
            while cc2.empty == True:
                n = n+1
                intervals = ufc.make_uf_nn_interval(nn, uf, F[:5], X_test[t:t+1])
                cc2, cfsexp2 = ufc.Triple_F(X, X_test[t:t+1], protectdf, F, catf, numf, intervals, features, bb, desired_outcome)
                if n >= 10:
                    break
            if cc2.empty == True:
                cnn = nn[n:n+1].values
                threeF_cfdf = pd.concat([threeF_cfdf, nn[n:n+1]], ignore_index=True, axis=0)
            threeF_cfdf = pd.concat([threeF_cfdf, cc2[:1]], ignore_index=True, axis=0)
            #threeF_cfdf.to_csv(r'C:\Users\~\plots\ufc-dice\threeF_mlp'+'.csv')
    end = time.time()
    threetime = end-start
    if len(X_test) != 0:
        threetime = threetime / len(X_test)
    print('ufce3 time:', threetime)
    return threeF_cfdf, threetime, foundidx, intervald, testout

def ufce_diverse(X, data_lab1, X_test, uf, F, numf, catf, features,f2change, protectdf, bb, desired_outcome, k):
    """
    :param X:
    :param data_lab1:
    :param X_test:
    :param uf:
    :param F: feature pairs
    :param numf:
    :param catf:
    :param features:
    :param f2change:
    :param protectdf:
    :param bb: black box
    :param desired_outcome:
    :param k:
    :return:
    """
    start = time.time()
    perturb_step = {}
    c = pd.DataFrame()
    nn, idx = ufc.NNkdtree(data_lab1, X_test, 100)  # increase radius size as per the dataset
    if nn.empty != True:
        interval1 = ufc.make_intervals(nn, uf, f2change, X_test)
        interval2 = ufc.make_uf_nn_interval(nn, uf, F[:5], X_test)
        t1 = ufc.Single_F(X_test, catf, interval1, bb, desired_outcome)
        if t1.empty != True:
            c = pd.concat([c, t1[:1]], ignore_index=True, axis=0)
        t2, explore2 = ufc.Double_F(X, X_test, protectdf, F[:5], catf, numf, interval2, features, bb, desired_outcome)
        if t2.empty != True:
            c = pd.concat([c, t2[:1]], ignore_index=True, axis=0)
            t2d, explore2d = ufc.Double_F(X, X_test, protectdf, F[2:5], catf, numf,
                                                                     interval2, features, bb,
                                                                     desired_outcome)
            if t2d.empty != True:
                flag = check_two_different_feature_changes(X_test, t2, t2d)
                if flag:
                    c = pd.concat([c, t2d[:1]], ignore_index=True, axis=0)
        t3, explore3 = ufc.Triple_F(X, X_test, protectdf, F[:5], catf, numf, interval2, features, bb, desired_outcome)
        if t3.empty != True:
            c = pd.concat([c, t3[:1]], ignore_index=True, axis=0)
    else:
        print(f' The neighbourhood is too small to find diverse counterfactuals.')
    return c, interval1, interval2

def diverse_cfs(data_lab1, X_test, test_instance, n):
    """
    :param data_lab1:
    :param X_test:
    :param test_instance:
    :param n:
    :return:
    """
    df, idx = ufc.NNkdtree(data_lab1, X_test, 100)
    diff = (df.astype(float) - test_instance).abs().sum(axis=1)
    # sort the data points by the number of feature changes
    diff_sorted = diff.sort_values()
    # return the top n data points with the least number of feature changes
    return df.loc[diff_sorted.index[:n]]

def check_two_different_feature_changes(data_point1, data_point2, other_data_point):
    """
    :param data_point1: cf1
    :param data_point2: cf2
    :param other_data_point: test instance
    :return:
    """
    num_changes1 = sum(1 for f1, f2 in zip(data_point1, other_data_point) if f1 != f2)
    num_changes2 = sum(1 for f1, f2 in zip(data_point2, other_data_point) if f1 != f2)
    different_features1 = set(i for i, (a, b) in enumerate(zip(data_point1, other_data_point)) if a != b)
    different_features2 = set(i for i, (a, b) in enumerate(zip(data_point2, other_data_point)) if a != b)
    num_different_features = len(different_features1.union(different_features2))
    return num_changes1 >= 2 and num_changes2 >= 2 and num_different_features >= 2

