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


import pandas as pd
import numpy as np



# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to find the row with the least Euclidean distance for specified continuous features
def find_best_row(df, test_instance, continuous_features):
    df['proximity'] = df.apply(lambda row: euclidean_distance(
        row[continuous_features].values, 
        test_instance[continuous_features].values), axis=1)

    best_row = df.loc[df['proximity'].idxmin()]
    return best_row

def changes_per_cf(x, cf):
    features = (list(x.columns))
    nbr_changes = 0
    for j in features:
        if cf[j].values != x[j].values:
            nbr_changes += 1
    return nbr_changes

def feasible_2method(test, cflist, changeable_features, uf):
    temp = pd.DataFrame()
    features = (list(test.columns))
    for x in range(len(cflist)):
        # plaus = int(ufc.lofn(test, cflist[x:x + 1], X_train[:], scaler))
        # if plaus == 1:
        nbr_act, f_list = ufc.nbr_actionable_cfn(test, cflist[x:x + 1], features, changeable_features)
        spar_count = changes_per_cf(test, cflist[x:x + 1])
        if nbr_act/spar_count >= 0.2: #for movie 5 or more, for bank=3, etc
            count_in = 0
            for j in f_list:
                limit = test[j][0] + uf[j]
                # print("in Actionability", X_test.at[x, j], uf[j])
                if cflist.at[x, j] <= limit:
                    count_in += 1
            if nbr_act == count_in:
                temp = pd.concat([temp, cflist[x:x + 1]], axis=0, ignore_index=True)
    return temp

def dice_cfexp(df, X_test, numf, f2change, no_cf, bb, uf, outcome_label):
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
    cf = pd.DataFrame()
    flag = 0
    idx = []
    for x in range(len(X_test)):
        # e1_v = exp.generate_counterfactuals(X_test[x:x+1], total_CFs=no_cf, desired_class="opposite", features_to_vary= f2change)#, permitted_range=uf_ranges
        # #e1.cf_examples_list[0].final_cfs_df[0:1].to_csv(path + '' + str(x) + '' + '.csv', index=False)
        # cf_v = e1_v.cf_examples_list[0].final_cfs_df[0:no_cf]
        # dice_cfs = pd.concat([dice_cfs_v, cf_v], ignore_index=True, axis=0)
        # idx.append(x)
        #uf_ranges = {'Income': [X_test.at[x, 'Income'], X_test.at[x, 'Income']+uf['Income']], 'CCAvg': [X_test.at[x, 'CCAvg'], X_test.at[x, 'CCAvg']+uf['CCAvg']], 'Mortgage': [X_test.at[x, 'Mortgage'], X_test.at[x, 'Mortgage']+uf['Mortgage']]}
        #uf_ranges = {'Income': [X_test.at[x, 'Income'], uf['Income']], 'CCAvg': [X_test.at[x, 'CCAvg'], uf['CCAvg']], 'Mortgage': [X_test.at[x, 'Mortgage'], uf['Mortgage']]} #'CDAccount': [X_test.at[x, 'CDAccount'], uf['CDAccount']],, 'Online': [X_test.at[x, 'Online'], uf['Online']] 'Education': [X_test[x:x + 1]['Education'][0], uf['Education']], 'CDAccount': [X_test[x:x + 1]['CDAccount'][0],  'Online': [X_test[x:x + 1]['Online'][0], uf['Online']]
        #uf_ranges = {'Mcv': [X_test.at[x, 'Mcv'], X_test.at[x, 'Mcv']+uf['Mcv']], 'Alkphos':[X_test.at[x, 'Alkphos'], X_test.at[x, 'Alkphos']+uf['Alkphos']], 'Sgpt':[X_test.at[x, 'Sgpt'], X_test.at[x, 'Sgpt']+uf['Sgpt']], 'Sgot':[X_test.at[x, 'Sgot'], uf['Sgot']], 'Gammagt':[X_test.at[x, 'Gammagt'], uf['Gammagt']], 'Drinks':[X_test.at[x, 'Drinks'], uf['Drinks']]}
        #uf_ranges = {'GRE Score': [X_test.at[x, 'GRE Score'], X_test.at[x, 'GRE Score']+uf['GRE Score']], 'TOEFL Score':[X_test.at[x, 'TOEFL Score'], X_test.at[x, 'TOEFL Score']+uf['TOEFL Score']], 'University Rating':[X_test.at[x, 'University Rating'], X_test.at[x, 'University Rating']+uf['University Rating']], 'SOP':[X_test.at[x, 'SOP'], X_test.at[x, 'SOP']+uf['SOP']], 'LOR':[X_test.at[x, 'LOR'], X_test.at[x, 'LOR']+uf['LOR']], 'CGPA':[X_test.at[x, 'CGPA'], X_test.at[x, 'CGPA']+uf['CGPA']]}
        #uf_ranges = {'fixed acidity': [X_test.at[x, 'fixed acidity'], X_test.at[x, 'fixed acidity']+uf['fixed acidity']], 'residual sugar':[X_test.at[x, 'residual sugar'], X_test.at[x, 'residual sugar']+uf['residual sugar']], 'free sulfur dioxide':[X_test.at[x, 'free sulfur dioxide'], X_test.at[x, 'free sulfur dioxide']+uf['free sulfur dioxide']], 'total sulfur dioxide':[X_test.at[x, 'total sulfur dioxide'], X_test.at[x, 'total sulfur dioxide']+uf['total sulfur dioxide']], 'pH':[X_test.at[x, 'pH'], X_test.at[x, 'pH']+uf['pH']], 'alcohol':[X_test.at[x, 'alcohol'], X_test.at[x, 'alcohol']+uf['alcohol']]}
        #uf_ranges = {'fixed acidity': [X_test.at[x, 'fixed acidity'], uf['fixed acidity']], 'residual sugar':[X_test.at[x, 'residual sugar'], uf['residual sugar']], 'free sulfur dioxide':[X_test.at[x, 'free sulfur dioxide'], uf['free sulfur dioxide']], 'total sulfur dioxide':[X_test.at[x, 'total sulfur dioxide'], uf['total sulfur dioxide']], 'pH':[X_test.at[x, 'pH'], uf['pH']], 'alcohol':[X_test.at[x, 'alcohol'], uf['alcohol']]}
        #uf_ranges = {'Production expense': [X_test.at[x, 'Production expense'], X_test.at[x, 'Production expense']+uf['Production expense']], 'Multiplex coverage':[X_test.at[x, 'Multiplex coverage'], X_test.at[x, 'Multiplex coverage']+uf['Multiplex coverage']], 'Num_multiplex':[X_test.at[x, 'Num_multiplex'], X_test.at[x, 'Num_multiplex']+uf['Num_multiplex']], 'Movie_length':[X_test.at[x, 'Movie_length'], uf['Movie_length']], 'Lead_ Actor_Rating':[X_test.at[x, 'Lead_ Actor_Rating'], uf['Lead_ Actor_Rating']], 'Lead_Actress_rating':[X_test.at[x, 'Lead_Actress_rating'], uf['Lead_Actress_rating']],'Director_rating':[X_test.at[x, 'Director_rating'], uf['Director_rating']], 'Producer_rating':[X_test.at[x, 'Producer_rating'], uf['Producer_rating']]}
        #sorted_uf = {key: sorted(value) for key, value in uf_ranges.items()}
        try:
            e1 = exp.generate_counterfactuals(X_test[x:x+1], total_CFs=no_cf, desired_class="opposite", features_to_vary= f2change, permitted_range=None)#, permitted_range=uf_ranges
            #e1.cf_examples_list[0].final_cfs_df[0:1].to_csv(path + '' + str(x) + '' + '.csv', index=False)
            foundcfs = e1.cf_examples_list[0].final_cfs_df[0:no_cf]
            flag = 1
            if len(foundcfs) > 1:
                # Find the best row based on Euclidean distance for the specified continuous features
                # foundcfs_f = feasible_2method(X_test[x:x+1], foundcfs, f2change, uf)
                best_row = find_best_row(foundcfs, X_test[x:x+1], numf)
                # Convert the best row to a DataFrame
                cf = best_row.to_frame().T
                # print("all cfs", foundcfs)
                # print("best cf",cf)
        except Exception as e:
            print(f"Error: {e}")
        if flag != 0:
            dice_cfs = pd.concat([dice_cfs, cf], ignore_index = True, axis = 0)
            flag = 0
            idx.append(x)
    #dice_cfs.to_csv(r'C:\Users\~plots\ufc-dice\DiCE_mlp'+'.csv')
    if len(dice_cfs) != 0:
        flag = 1
    end = time.time()
    dicetime = end-start
    if len(X_test) != 0:
        dicetime = dicetime/len(X_test)
    print('\t\t dice time:', dicetime)
    return dice_cfs, idx, dicetime, flag

def dice_cfexp_in(df, X_test, numf, f2change, no_cf, bb, uf, outcome_label):
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
    cf = pd.DataFrame()
    flag = 0
    idx = []
    for x in range(len(X_test)):
        # e1_v = exp.generate_counterfactuals(X_test[x:x+1], total_CFs=no_cf, desired_class="opposite", features_to_vary= f2change)#, permitted_range=uf_ranges
        # #e1.cf_examples_list[0].final_cfs_df[0:1].to_csv(path + '' + str(x) + '' + '.csv', index=False)
        # cf_v = e1_v.cf_examples_list[0].final_cfs_df[0:no_cf]
        # dice_cfs = pd.concat([dice_cfs_v, cf_v], ignore_index=True, axis=0)
        # idx.append(x)
        # uf_ranges = {'Income': [X_test.at[x, 'Income'], X_test.at[x, 'Income']+uf['Income']], 'CCAvg': [X_test.at[x, 'CCAvg'], X_test.at[x, 'CCAvg']+uf['CCAvg']], 'Mortgage': [X_test.at[x, 'Mortgage'], X_test.at[x, 'Mortgage']+uf['Mortgage']]}
        # uf_ranges = {'Income': [X_test.at[x, 'Income'], uf['Income']], 'CCAvg': [X_test.at[x, 'CCAvg'], uf['CCAvg']], 'Mortgage': [X_test.at[x, 'Mortgage'], uf['Mortgage']]} #'CDAccount': [X_test.at[x, 'CDAccount'], uf['CDAccount']],, 'Online': [X_test.at[x, 'Online'], uf['Online']] 'Education': [X_test[x:x + 1]['Education'][0], uf['Education']], 'CDAccount': [X_test[x:x + 1]['CDAccount'][0],  'Online': [X_test[x:x + 1]['Online'][0], uf['Online']]
        # uf_ranges = {'Mcv': [X_test.at[x, 'Mcv'], X_test.at[x, 'Mcv']+uf['Mcv']], 'Alkphos':[X_test.at[x, 'Alkphos'], X_test.at[x, 'Alkphos']+uf['Alkphos']], 'Sgpt':[X_test.at[x, 'Sgpt'], X_test.at[x, 'Sgpt']+uf['Sgpt']], 'Sgot':[X_test.at[x, 'Sgot'], uf['Sgot']], 'Gammagt':[X_test.at[x, 'Gammagt'], uf['Gammagt']], 'Drinks':[X_test.at[x, 'Drinks'], uf['Drinks']]}
        # uf_ranges = {'GRE Score': [X_test.at[x, 'GRE Score'], X_test.at[x, 'GRE Score'] + uf['GRE Score']], 'TOEFL Score': [X_test.at[x, 'TOEFL Score'], X_test.at[x, 'TOEFL Score'] + uf['TOEFL Score']], 'University Rating': [X_test.at[x, 'University Rating'], X_test.at[x, 'University Rating'] + uf['University Rating']], 'SOP': [X_test.at[x, 'SOP'], X_test.at[x, 'SOP'] + uf['SOP']], 'LOR': [X_test.at[x, 'LOR'], X_test.at[x, 'LOR'] + uf['LOR']], 'CGPA': [X_test.at[x, 'CGPA'], X_test.at[x, 'CGPA'] + uf['CGPA']]}
        uf_ranges = {'GRE Score': [X_test.at[x, 'GRE Score'], uf['GRE Score']], 'TOEFL Score':[X_test.at[x, 'TOEFL Score'], uf['TOEFL Score']], 'University Rating':[X_test.at[x, 'University Rating'], uf['University Rating']], 'SOP':[X_test.at[x, 'SOP'], uf['SOP']], 'LOR':[X_test.at[x, 'LOR'], uf['LOR']], 'CGPA':[X_test.at[x, 'CGPA'], uf['CGPA']]}
        # uf_ranges = {'fixed acidity': [X_test.at[x, 'fixed acidity'], X_test.at[x, 'fixed acidity']+uf['fixed acidity']], 'residual sugar':[X_test.at[x, 'residual sugar'], X_test.at[x, 'residual sugar']+uf['residual sugar']], 'free sulfur dioxide':[X_test.at[x, 'free sulfur dioxide'], X_test.at[x, 'free sulfur dioxide']+uf['free sulfur dioxide']], 'total sulfur dioxide':[X_test.at[x, 'total sulfur dioxide'], X_test.at[x, 'total sulfur dioxide']+uf['total sulfur dioxide']], 'pH':[X_test.at[x, 'pH'], X_test.at[x, 'pH']+uf['pH']], 'alcohol':[X_test.at[x, 'alcohol'], X_test.at[x, 'alcohol']+uf['alcohol']]}
        # uf_ranges = {'fixed acidity': [X_test.at[x, 'fixed acidity'], uf['fixed acidity']], 'residual sugar':[X_test.at[x, 'residual sugar'], uf['residual sugar']], 'free sulfur dioxide':[X_test.at[x, 'free sulfur dioxide'], uf['free sulfur dioxide']], 'total sulfur dioxide':[X_test.at[x, 'total sulfur dioxide'], uf['total sulfur dioxide']], 'pH':[X_test.at[x, 'pH'], uf['pH']], 'alcohol':[X_test.at[x, 'alcohol'], uf['alcohol']]}
        # uf_ranges = {'Production expense': [X_test.at[x, 'Production expense'], X_test.at[x, 'Production expense'] + uf['Production expense']], 'Multiplex coverage': [X_test.at[x, 'Multiplex coverage'], X_test.at[x, 'Multiplex coverage'] + uf['Multiplex coverage']], 'Num_multiplex': [X_test.at[x, 'Num_multiplex'], X_test.at[x, 'Num_multiplex'] + uf['Num_multiplex']], 'Movie_length': [X_test.at[x, 'Movie_length'], uf['Movie_length']], 'Lead_ Actor_Rating': [X_test.at[x, 'Lead_ Actor_Rating'], uf['Lead_ Actor_Rating']], 'Lead_Actress_rating': [X_test.at[x, 'Lead_Actress_rating'], uf['Lead_Actress_rating']], 'Director_rating': [X_test.at[x, 'Director_rating'], uf['Director_rating']], 'Producer_rating': [X_test.at[x, 'Producer_rating'], uf['Producer_rating']]}
        # uf_ranges = {'Production expense': [X_test.at[x, 'Production expense'], uf['Production expense']], 'Multiplex coverage':[X_test.at[x, 'Multiplex coverage'], uf['Multiplex coverage']], 'Num_multiplex':[X_test.at[x, 'Num_multiplex'], uf['Num_multiplex']], 'Movie_length':[X_test.at[x, 'Movie_length'], uf['Movie_length']], 'Lead_ Actor_Rating':[X_test.at[x, 'Lead_ Actor_Rating'], uf['Lead_ Actor_Rating']], 'Lead_Actress_rating':[X_test.at[x, 'Lead_Actress_rating'], uf['Lead_Actress_rating']],'Director_rating':[X_test.at[x, 'Director_rating'], uf['Director_rating']], 'Producer_rating':[X_test.at[x, 'Producer_rating'], uf['Producer_rating']]}
        sorted_uf = {key: sorted(value) for key, value in uf_ranges.items()}
        try:
            e1 = exp.generate_counterfactuals(X_test[x:x+1], total_CFs=no_cf, desired_class="opposite", features_to_vary= f2change, permitted_range=sorted_uf)#, permitted_range=uf_ranges
            #e1.cf_examples_list[0].final_cfs_df[0:1].to_csv(path + '' + str(x) + '' + '.csv', index=False)
            foundcfs = e1.cf_examples_list[0].final_cfs_df[0:no_cf]
            flag = 1
            if len(foundcfs) > 1:
                # Find the best row based on Euclidean distance for the specified continuous features
                # foundcfs_f = feasible_2method(X_test[x:x+1], foundcfs, f2change, uf)
                best_row = find_best_row(foundcfs, X_test[x:x+1], numf)
                # Convert the best row to a DataFrame
                cf = best_row.to_frame().T
                # print("all cfs", foundcfs)
                # print("best cf",cf)
        except Exception as e:
            print(f"Error: {e}")
        if flag != 0:
            dice_cfs = pd.concat([dice_cfs, cf], ignore_index = True, axis = 0)
            flag = 0
            idx.append(x)
    #dice_cfs.to_csv(r'C:\Users\~plots\ufc-dice\DiCE_mlp'+'.csv')
    if len(dice_cfs) != 0:
        flag = 1
    end = time.time()
    dicetime = end-start
    if len(X_test) != 0:
        dicetime = dicetime/len(X_test)
    print('\t\t dice time:', dicetime)
    return dice_cfs, idx, dicetime, flag

def ar_cfexp(X, numf, bb, X_test, uf, scaler, X_train, f2change):
    """
    :param X: X data
    :param numf: numerical features
    :param bb: blackbox model
    :param X_test: test set
    :return ar_cfs: ar counterfactuals
    """
    start = time.time()
    flag = 0
    idx = []
    A = rs.ActionSet(X)
    from IPython.core.display import display, HTML
    clf = bb
    A.set_alignment(clf)
    finalarcfs = pd.DataFrame()
    cf = pd.DataFrame()
    for x in range(len(X_test)):
        # uf_ranges = {'Income': [X_test.at[x, 'Income'], uf['Income']], 'CCAvg': [X_test.at[x, 'CCAvg'], uf['CCAvg']], 'Mortgage': [X_test.at[x, 'Mortgage'], uf['Mortgage']], 'CDAccount': [X_test.at[x, 'CDAccount'], uf['CDAccount']], 'Online': [X_test.at[x, 'Online'], uf['Online']]} #, 'Education': [X_test[x:x + 1]['Education'][0], uf['Education']], 'CDAccount': [X_test[x:x + 1]['CDAccount'][0],  'Online': [X_test[x:x + 1]['Online'][0], uf['Online']]
        # sorted_uf = {key: sorted(value) for key, value in uf_ranges.items()}
        # print(sorted_uf)
        try:
            fs = rs.Flipset(X_test[x:x + 1].values, action_set=A, clf=clf)
            # yet to generate more cfs by chaging total_items 10, 5, or etc.
            fs.populate(enumeration_type='distinct_subsets', total_items = 5) #'distinct_subsets', 'mutually_exclusive' for wine datset use mutually exclusive, for rest distinct
            f_list = numf #['Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
            #features that need to be changed and could flip the outcome bt fs
            candi_cfs = pd.DataFrame()
            # finding k_cfs 
            if len(fs) > 1:
                for i in range(len(fs)):
                    # print(fs.df['features'][i])
                    # print(fs.df['x_new'][i])
                    feat2change = fs.df['features'][i]
                    values_2change = fs.df['x_new'][i]
                    changed_instance = X_test[x:x+1].copy()
                    for f, i in enumerate(feat2change):
                        changed_instance[i] = values_2change[f]
                    candi_cfs = pd.concat([candi_cfs, changed_instance], ignore_index=True, axis=0)
                # print("AR-1, candidates", candi_cfs)
                # filering best cf
                # feas_cfs = feasible_2method(X_test[x:x+1], candi_cfs, X_train, scaler, list(X_test[x:x+1].columns), f2change, uf)
                # print("AR-1, feasibles,", feas_cfs)
                if len(candi_cfs) > 1:
                    # foundcfs_f = feasible_2method(X_test[x:x+1], candi_cfs, f2change, uf)
                    best_row = find_best_row(candi_cfs, X_test[x:x+1], numf)
                    cf = best_row.to_frame().T
                    idx.append(x)
                    finalarcfs = pd.concat([finalarcfs, cf], ignore_index=True, axis=0)
                    finalarcfs = finalarcfs.drop(['proximity'], axis=1)
                    # print("AR-1, found best", cf)
            else:
                # print("AR, else, only one")
                feat2change = fs.df['features']
                values_2change = fs.df['x_new']
                changed_instance = X_test[x:x+1].copy()
                for f, i in enumerate(feat2change):
                    changed_instance[i] = values_2change[f]
                idx.append(x)
                finalarcfs = pd.concat([finalarcfs, changed_instance], ignore_index=True, axis=0)
        except Exception as e:
            print(f"Error: {e}")
    # arcfs = pd.DataFrame()
    # for x, row in finalarcfs.iterrows():
    #     uf_ranges = {'Income': [X_test.at[x, 'Income'], uf['Income']], 'CCAvg': [X_test.at[x, 'CCAvg'], uf['CCAvg']], 'Mortgage': [X_test.at[x, 'Mortgage'], uf['Mortgage']], 'CDAccount': [X_test.at[x, 'CDAccount'], uf['CDAccount']], 'Online': [X_test.at[x, 'Online'], uf['Online']]} #, 'Education': [X_test[x:x + 1]['Education'][0], uf['Education']], 'CDAccount': [X_test[x:x + 1]['CDAccount'][0],  'Online': [X_test[x:x + 1]['Online'][0], uf['Online']]
    #     sorted_uf = {key: sorted(value) for key, value in uf_ranges.items()}
    #     mask = all(start_range <= row[feature] <= end_range for feature, (start_range, end_range) in sorted_uf.items())
    #     if mask:
    #         filtered_row = finalarcfs.loc[x]
    #         filtered_row = pd.DataFrame(filtered_row).transpose()
    #         arcfs = pd.concat([arcfs, filtered_row], ignore_index=True, axis=0)
    #         flag = 1
    # print("AR count of actionable", len(arcfs))
    end = time.time()
    artime = end-start
    if len(X_test) != 0:
        artime = artime / len(X_test)
    # if len(arcfs) != 0:
    #     flag = 1
    print('\t\t AR time:', artime)
    return finalarcfs, artime, idx

def sfexp(X, data_lab1, X_test, uf, step, f2change, numf, catf, bb, desired_outcome, k, order):
    """
    :param X:
    :param data_lab1:
    :param X_test:
    :param uf:
    :param step:
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
    cf = pd.DataFrame()
    found_indexes = []
    intervald = dict()
    start = time.time()
    for t in range(len(X_test)):
        n = 0
        nn, idx = ufc.NNkdtree(data_lab1, X_test[t:t+1], 500) #increase radius size as per the dataset, for movie 5k, rest 100 and 500
        if nn.empty != True:
            # print("here sf")
            interval = ufc.make_intervals(nn, uf, f2change, X_test[t:t+1])
            cc = ufc.Single_F(X_test[t:t+1], catf, interval, bb, desired_outcome, step)
            if cc.empty != True:
                if len(cc) > 1:
                    # foundcfs_f = feasible_2method(X_test[t:t+1], cc, f2change, uf)
                    best_row = find_best_row(cc, X_test[t:t+1], numf)
                    cf = best_row.to_frame().T
                    found_indexes.append(t)
                    oneF_cfdf = pd.concat([oneF_cfdf, cf], ignore_index=True, axis=0)
                    oneF_cfdf = oneF_cfdf.drop(['proximity'], axis=1)
                else:
                    found_indexes.append(t)
                    # intervald[t] = interval
                    oneF_cfdf = pd.concat([oneF_cfdf, cc[:1]], ignore_index=True, axis=0)
                    # testout = pd.concat([testout, X_test[t:t+1]], ignore_index=True, axis=0)
            
            # # to find the nearest optimal CF
            # distances = np.linalg.norm(cc.values - X_test[t:t+1].values, axis=1)
            # nearest_row_index = np.argmin(distances)
            # nearest_cf = cc.iloc[[nearest_row_index]]
            # while cc.empty == True:
            #     n = n+1
            #     interval = ufc.make_intervals(nn, uf, f2change, X_test[t:t+1])
            #     cc = ufc.Single_F(X_test[t:t+1], catf, interval, bb, desired_outcome, step)
            #     if n >= 10:
            #         break
            # if cc.empty == True:
            #     oneF_cfdf = pd.concat([oneF_cfdf, nn[0:1]], ignore_index=True, axis=0)
            #     found_indexes.append(t)
        #onetest_cfdf = pd.concat([onetest_cfdf, test[:1]], ignore_index=True, axis=0)
        #oneF_cfdf.to_csv(r'C:\Users\~\plots\ufc-dice\OneF_mlp'+'.csv')
    end = time.time()
    onetime = end - start
    print('\t\t ufce1 time', onetime)
    if len(X_test) != 0:
        onetime = onetime/len(X_test)
    return oneF_cfdf, onetime, found_indexes #, intervald, testout

def dfexp(X, data_lab1, X_test, uf, F, numf, catf, features, protectf, bb, desired_outcome, k, order):
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
        nn, idx = ufc.NNkdtree(data_lab1, X_test[t:t+1], 500)
        if nn.empty != True:
            intervals = ufc.make_uf_nn_interval(nn, uf, F[:], X_test[t:t+1])
            # print("here nn interval", intervals)
            cc2, cfsexp2 = ufc.Double_F(X, X_test[t:t+1], protectedf, F[:], catf, numf, intervals, features, bb, desired_outcome, order, k)
            # print(f"cf count for test{t}", len(cc2))
            if cc2.empty != True:
                if len(cc2) > 1:
                    # foundcfs_f = feasible_2method(X_test[t:t+1], cc2, f2change, uf)
                    best_row = find_best_row(cc2, X_test[t:t+1], numf)
                    cf = best_row.to_frame().T
                    foundidx.append(t)
                    twoF_cfdf = pd.concat([twoF_cfdf, cf], ignore_index=True, axis=0)
                    twoF_cfdf = twoF_cfdf.drop(['proximity'], axis=1)
                else:
                    foundidx.append(t)
                    # intervald[t] = interval
                    twoF_cfdf = pd.concat([twoF_cfdf, cc2[:1]], ignore_index=True, axis=0)
                    # testout = pd.concat([testout, X_test[t:t+1]], ignore_index=True, axis=0)
                # foundidx.append(t)
                # df2 = cc2.iloc[-1:]
                # twoF_cfdf = pd.concat([twoF_cfdf, df2], ignore_index=True, axis=0)
                # intervald[t] = intervals
                # testout = pd.concat([testout, X_test[t:t + 1]], ignore_index=True, axis=0)
            else:
                # print("length of cfexp", len(cfsexp2))
                if cfsexp2.empty != True:
                    predictions = bb.predict(cfsexp2)
                    selected_rows = cfsexp2[predictions == desired_outcome]
                    if selected_rows.empty != True:
                        # print("found in cfexp")
                        # print(selected_rows[:1])
                        twoF_cfdf = pd.concat([twoF_cfdf, selected_rows[:1]], ignore_index=True, axis=0)

            
            # # to find the nearest optimal CF
            # distances = np.linalg.norm(cc2.values - X_test[t:t+1].values, axis=1)
            # nearest_row_index = np.argmin(distances)
            # nearest_cf = cc2.iloc[[nearest_row_index]]
            # while cc2.empty == True:
            #     n = n+1
            #     intervals = ufc.make_uf_nn_interval(nn, uf, F[:5], X_test[t:t+1])
            #     cc2, cfsexp2 = ufc.Double_F(X, X_test[t:t+1], protectedf, F[:5], catf, numf, intervals, features, bb, desired_outcome)
            #     if n >= 10:
            #         break
            # if cc2.empty == True:
            #     twoF_cfdf = pd.concat([twoF_cfdf, nn[0:1]], ignore_index=True, axis=0)
            #     foundidx.append(t)
            # twoF_cfdf = pd.concat([twoF_cfdf, cc2[:1]], ignore_index=True, axis=0)
            #twoF_cfdf.to_csv(r'C:\Users\~\plots\ufc-dice\twoF'+'.csv')
    end = time.time()
    twotime = end-start
    if len(X_test) != 0:
        twotime = twotime/len(X_test)
    print('\t\t ufce2 time:', twotime)
    return twoF_cfdf, twotime, foundidx #, intervald, testout

def tfexp(X, data_lab1, X_test, uf, F, numf, catf, feature2change, protectdf, bb, desired_outcome, k, order):
    """
    :param X:
    :param data_lab1:
    :param X_test:
    :param uf:
    :param F:
    :param numf:
    :param catf:
    :param feature2change:
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
        nn, idx = ufc.NNkdtree(data_lab1, X_test[t:t+1], 500)
        if nn.empty != True:
            # print("here tf")
            intervals = ufc.make_uf_nn_interval(nn, uf, F[:], X_test[t:t+1]) 
            cc3, cfsexp2 = ufc.Triple_F(X, X_test[t:t+1], protectdf, F[:], catf, numf, intervals, feature2change, bb, desired_outcome, order, k) #features
            # print("interval", intervals)
            # print("test", X_test[t:t + 1].values)
            # print("cf", cc3[:1].values)
            if cc3.empty != True:
                if len(cc3) > 1:
                    # foundcfs_f = feasible_2method(X_test[t:t+1], cc3, f2change, uf)
                    best_row = find_best_row(cc3, X_test[t:t+1], numf)
                    cf = best_row.to_frame().T
                    foundidx.append(t)
                    threeF_cfdf = pd.concat([threeF_cfdf, cf], ignore_index=True, axis=0)
                    threeF_cfdf = threeF_cfdf.drop(['proximity'], axis=1)
                else:
                    foundidx.append(t)
                    # intervald[t] = interval
                    threeF_cfdf = pd.concat([threeF_cfdf, cc3[:1]], ignore_index=True, axis=0)
                
                
                # foundidx.append(t)
                # threeF_cfdf = pd.concat([threeF_cfdf, cc3[:1]], ignore_index=True, axis=0)
            # # to find the nearest optimal CF
            # distances = np.linalg.norm(cc3.values - X_test[t:t+1].values, axis=1)
            # nearest_row_index = np.argmin(distances)
            # nearest_cf = cc3.iloc[[nearest_row_index]]

            #     intervald[t] = intervals
            #     testout = pd.concat([testout, X_test[t:t + 1]], ignore_index=True, axis=0)
            # while cc2.empty == True:
            #     n = n+1
            #     intervals = ufc.make_uf_nn_interval(nn, uf, F[:5], X_test[t:t+1])
            #     cc2, cfsexp2 = ufc.Triple_F(X, X_test[t:t+1], protectdf, F[:5], catf, numf, intervals, feature2change, bb, desired_outcome) #features
            #     if n >= 10:
            #         break
            # if cc2.empty == True:
            #     cnn = nn[n:n+1].values
            #     threeF_cfdf = pd.concat([threeF_cfdf, nn[0:1]], ignore_index=True, axis=0)
            #     foundidx.append(t)
            # threeF_cfdf = pd.concat([threeF_cfdf, cc2[:1]], ignore_index=True, axis=0)
            #threeF_cfdf.to_csv(r'C:\Users\~\plots\ufc-dice\threeF_mlp'+'.csv')
    end = time.time()
    threetime = end-start
    if len(X_test) != 0:
        threetime = threetime / len(X_test)
    print('ufce3 time:', threetime)
    return threeF_cfdf, threetime, foundidx #, intervald, testout

