import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import sklearn.neural_network as sknet
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

def get_model_lr():
    lg_bank = pickle.load(open('lg_model_bank.sav', 'rb'))
    lg_credit = pickle.load(open('lg_model_credit.sav', 'rb'))
    lg_adult = pickle.load(open('lg_model_adult.sav', 'rb'))
    lg_append = pickle.load(open('lg_model_appendicitis.sav', 'rb'))
    lg_bupa = pickle.load(open('lg_model_bupa.sav', 'rb'))
    lg_heart = pickle.load(open('lg_model_heart.sav', 'rb'))
    #lg_magic = pickle.load(open('lg_model_magic.sav', 'rb'))
    lg_movie = pickle.load(open('lg_model_movie.sav', 'rb'))
    lg_mammo = pickle.load(open('lg_model_mammo.sav', 'rb'))
    lg_saheart = pickle.load(open('lg_model_saheart.sav', 'rb'))
    lg_spotify = pickle.load(open('lg_model_spotify.sav', 'rb'))
    lg_titanic = pickle.load(open('lg_model_titanic.sav', 'rb'))
    lg_wdbc = pickle.load(open('lg_model_wdbc.sav', 'rb'))
    lg_wine = pickle.load(open('lg_model_wine-red.sav', 'rb'))
    lg_wiscon = pickle.load(open('lg_model_wisconsin.sav', 'rb'))
    return lg_bank, lg_credit, lg_adult, lg_append, lg_bupa, lg_heart, lg_movie, lg_mammo, lg_saheart, lg_spotify, lg_titanic, lg_wdbc, lg_wine, lg_wiscon

def get_model_mlp():
    mlp_bank = pickle.load(open('mlp_model_bank.sav', 'rb'))
    mlp_credit = pickle.load(open('mlp_model_credit.pkl', 'rb'))
    mlp_adult = pickle.load(open('mlp_model_adult.sav', 'rb'))
    mlp_append = pickle.load(open('mlp_model_appendicitis.sav', 'rb'))
    mlp_bupa = pickle.load(open('mlp_model_bupa.sav', 'rb'))
    mlp_heart = pickle.load(open('mlp_model_heart.sav', 'rb'))
    # lg_magic = pickle.load(open('lg_model_magic.sav', 'rb'))
    mlp_movie = pickle.load(open('mlp_model_movie.sav', 'rb'))
    mlp_mammo = pickle.load(open('mlp_model_mammo.sav', 'rb'))
    mlp_saheart = pickle.load(open('mlp_model_saheart.sav', 'rb'))
    mlp_spotify = pickle.load(open('mlp_model_spotify.sav', 'rb'))
    mlp_titanic = pickle.load(open('mlp_model_titanic.sav', 'rb'))
    mlp_wdbc = pickle.load(open('mlp_model_wdbc.sav', 'rb'))
    mlp_wine = pickle.load(open('mlp_model_wine-red.sav', 'rb'))
    mlp_wiscon = pickle.load(open('mlp_model_wisconsin.sav', 'rb'))
    return mlp_bank, mlp_credit, mlp_adult,  mlp_append, mlp_bupa, mlp_heart, mlp_movie, mlp_mammo, mlp_saheart, mlp_spotify, mlp_titanic, mlp_wdbc, mlp_wine, mlp_wiscon

def classify_dataset_getModel(dataset_df, data_name=''):
    """
    :param dataset_df:
    :param data_name:
    :return:
    """
    if data_name == 'bank':
        dataset_df.reset_index(drop=True, inplace=True)
        del dataset_df['Unnamed: 0']
        del dataset_df['age']
        del dataset_df['Experience']
        X = dataset_df.loc[:, dataset_df.columns != 'Personal Loan']
        y = dataset_df['Personal Loan']
    elif data_name == 'grad':
        dataset_df.reset_index(drop=True, inplace=True)
        del dataset_df['Unnamed: 0']
        X = dataset_df.loc[:, dataset_df.columns != 'Chance of Admit']
        y = dataset_df['Chance of Admit']
    elif data_name == 'wine':
        dataset_df.reset_index(drop=True, inplace=True)
        del dataset_df['Unnamed: 0']
        X = dataset_df.loc[:, dataset_df.columns != 'quality']
        y = dataset_df['quality']
    elif data_name == 'movie':
        dataset_df.reset_index(drop=True, inplace=True)
        del dataset_df['Unnamed: 0']
        X = dataset_df.loc[:, dataset_df.columns != 'Start_Tech_Oscar']
        y = dataset_df['Start_Tech_Oscar']
    elif data_name == 'bupa':
        dataset_df.reset_index(drop=True, inplace=True)
        # del dataset_df['Unnamed: 0']
        X = dataset_df.loc[:, dataset_df.columns != 'Selector']
        y = dataset_df['Selector']
    elif data_name == 'adult':
        dataset_df.reset_index(drop=True, inplace=True)
        del dataset_df['Unnamed: 0']
        X = dataset_df.loc[:, dataset_df.columns != 'class']
        y = dataset_df['class']

    # train-test splits with a random state that provides the best distribution fit of data
    n_features = X.shape[1]
    n_tries = 10 # no. of iterations/tries to analyse the best random state of dataset
    result = []
    for random_state in range(n_tries):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)
        distances = list(
            map(lambda i: stats.ks_2samp(X_train.iloc[:, i], X_test.iloc[:, i]).statistic, range(n_features)))
        result.append((random_state, max(distances)))
    result.sort(key=lambda x: x[1])
    # from result, random state is important to train the model on the best split
    idx = 0
    random_state = result[idx][0]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)

    ## 10-fold cross validation

    mlp = MLPClassifier(max_iter=1000)
    mlp.fit(Xtrain, ytrain)
    scores1 = cross_val_score(mlp, X=Xtrain, y=ytrain, cv=10, n_jobs=1)
    mlp_mean, mlp_std = np.mean(scores1), np.std(scores1)
    # preds = mlp.predict(Xtest)
    # tn, fp, fn, tp = confusion_matrix(ytest, preds)
    # predictions = clf.predict(inputs)
    # for input, prediction, label in zip(inputs, predictions, labels):
    #     if prediction != label:
    #         print(input, 'has been classified as ', prediction, 'and should be ', label)
    # recal_mlp = tp / (tp+fn)
    # mlp_r2 = r2_score(ytest.values.ravel(), mlp.predict(Xtest))
    # mlp_acc = accuracy_score(ytest.values.ravel(), mlp.predict(Xtest))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtrain, ytrain)
    scores2 = cross_val_score(mlp, X=Xtrain, y=ytrain, cv=10, n_jobs=1)
    lr_mean, lr_std = np.mean(scores2), np.std(scores2)
    # lr_r2 = r2_score(ytest.values.ravel(), lr.predict(Xtest))
    # lr_acc = accuracy_score(ytest.values.ravel(), lr.predict(Xtest))

    return mlp, mlp_mean, mlp_std, lr, lr_mean, lr_std, Xtest, Xtrain, X, y, dataset_df

def get_bank_user_constraints(bankloan):
    """
    :param bankloan: bank dataframe
    :return:
    """
    features = ['Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
                'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    catf = ['SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
    numf = ['Income', 'Family', 'CCAvg', 'Education', 'Mortgage']
    uf = {'Income': 70, 'CCAvg': 3.0, 'Family': 3, 'Education': 3, 'Mortgage': 100, 'CDAccount': 1, 'Online': 1,
          'SecuritiesAccount': 1, 'CreditCard': 1}
    # uf  = getMCSvalues()
    f2change = ['Income', 'CCAvg', 'Mortgage', 'CDAccount', 'Online']
    outcome_label = 'Personal Loan'
    desired_outcome = 1.0
    nbr_features = 9
    protectf = []

    # desired space
    data_lab1 = pd.DataFrame()
    data_lab1 = bankloan[bankloan["Personal Loan"] == 1]
    data_lab0 = bankloan[bankloan["Personal Loan"] == 0]
    data_lab1 = data_lab1.drop(['Personal Loan'], axis=1)
    return features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1

def get_grad_user_constraints(grad):
    """
    :param grad:
    :return:
    """
    features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR', 'CGPA', 'Research']
    catf = ['Research']
    numf = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR', 'CGPA']
    uf = {'GRE Score':20, 'TOEFL Score':10, 'University Rating':3, 'SOP':2,
       'LOR':2, 'CGPA':5, 'Research':1}
    # uf  = getMCSvalues()
    f2change = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR', 'CGPA', 'Research']
    outcome_label = 'Chance of Admit'
    desired_outcome = 1.0
    nbr_features = 7
    protectf = []
    # desired space
    data_lab1 = pd.DataFrame()
    data_lab1 = grad[grad["Chance of Admit"] == 1]
    data_lab0 = grad[grad["Chance of Admit"] == 0]
    data_lab1 = data_lab1.drop(['Chance of Admit'], axis=1)
    return features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1

def get_wine_user_constraints(wine):
    """
    :param wine:
    :return:
    """
    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol']
    catf = []
    numf = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol']
    uf = {'fixed acidity':3.0, 'residual sugar':3.0, 'free sulfur dioxide':8.0, 'total sulfur dioxide':12.0, 'pH':1.0, 'alcohol':2.0, 'density':0.20, 'volatile acidity':0.20, 'citric acid':0.8}
    # uf  = getMCSvalues()
    f2change = ['fixed acidity', 'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'alcohol', 'density', 'volatile acidity']
    outcome_label = 'quality'
    desired_outcome = 1.0
    nbr_features = 11
    protectf = []
    # desired space
    data_lab1 = pd.DataFrame()
    data_lab1 = wine[wine["quality"] == 1]
    data_lab0 = wine[wine["quality"] == 0]
    data_lab1 = data_lab1.drop(['quality'], axis=1)
    return features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1

def get_bupa_user_constraints(bupa):
    """
    :param bupa:
    :return:
    """
    features = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks']
    catf = []
    numf = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks']
    uf = {'Mcv':20, 'Alkphos':15, 'Sgpt':15, 'Sgot':15, 'Gammagt':15, 'Drinks':2}
    # uf  = getMCSvalues()
    f2change = ['Sgpt', 'Sgot', 'Gammagt']
    outcome_label = 'Selector'
    desired_outcome = 1.0
    nbr_features = 6
    protectf = []
    # desired space
    data_lab1 = pd.DataFrame()
    data_lab1 = bupa[bupa["Selector"] == 1]
    data_lab0 = bupa[bupa["Selector"] == 2]
    data_lab1 = data_lab1.drop(['Selector'], axis=1)
    return features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1

def get_movie_user_constraints(movie):
    """
    :param movie:
    :return:
    """
    features = ['Marketing expense', 'Production expense', 'Multiplex coverage',
               'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',
               'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',
               '3D_available', 'Time_taken', 'Twitter_hastags', 'Genre',
               'Avg_age_actors', 'Num_multiplex', 'Collection']
    catf = ['3D_available']
    numf = ['Marketing expense', 'Production expense', 'Multiplex coverage',
               'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',
               'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views', 'Time_taken', 'Twitter_hastags', 'Genre',
               'Avg_age_actors', 'Num_multiplex', 'Collection']
    uf = {'Production expense': 40, 'Num_multiplex':50,'Multiplex coverage':0.4, 'Movie_length':35, 'Lead_ Actor_Rating':4.0, 'Lead_Actress_rating':4.0,
               'Director_rating':4.0, 'Producer_rating':4.0, 'Genre':3, 'Collection':20000}
    # uf  = getMCSvalues()
    f2change = ['Production expense', 'Multiplex coverage','Num_multiplex', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',
               'Director_rating', 'Producer_rating', 'Genre', 'Collection']
    outcome_label = 'Start_Tech_Oscar'
    desired_outcome = 1.0
    nbr_features = 18
    protectf = []
    # desired space
    data_lab1 = pd.DataFrame()
    data_lab1 = movie[movie["Start_Tech_Oscar"] == 1]
    data_lab0 = movie[movie["Start_Tech_Oscar"] == 2]
    data_lab1 = data_lab1.drop(['Start_Tech_Oscar'], axis=1)
    return features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1

def create_folds(data, path, n_folds=5): # take X, y, concat
    """
    :param data:
    :param path:
    :param n_folds:
    :return:
    """
    data = data.sample(frac=1)
    fold_size = int(data.shape[0] / n_folds)
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size
        fold_X_test = data[start:end]

        #fold_path = os.path.join(path, 'fold_{}'.format(i)) # this will write in specific fold
        # if not os.path.exists(fold_path):
        #     os.makedirs(fold_path)
        fold_X_test.to_csv(os.path.join(path, 'testfold_{}.csv'.format(i)), index=False) #writing files on open path

def predict_X_test_folds(model, read_path, write_path, outcome_label):
    """
    :param model:
    :param read_path:
    :param write_path:
    :param outcome_label:
    :return:
    """
    testfolds = glob.glob(os.path.join(read_path, "*.csv"))
    for i, fold in enumerate(testfolds):
        testset = pd.read_csv(fold)
        testset = testset.reset_index()
        ltest = len(testset)
        ytest = testset[outcome_label]
        totest = testset.copy()
        totest = totest.drop(["index"], axis = 1)
        totest = totest.drop([outcome_label], axis=1)
        preds = model.predict(totest)
        # get indices of correctly classified instances
        correct_indices = np.where(preds == ytest)[0]
        # get only the correctly classified instances
        X_test_correct = testset.loc[correct_indices]
        lxtest = len(X_test_correct)
        misclassified = ltest - lxtest
        if outcome_label=='Selector':
            X_test_pred_0 = X_test_correct[X_test_correct[outcome_label] == 1]
        else:
            X_test_pred_0 = X_test_correct[X_test_correct[outcome_label] == 0]
        del X_test_pred_0['index']
        del X_test_pred_0[outcome_label]
        if len(X_test_pred_0) >= 50:
            X_test_pred_0[:50].to_csv(os.path.join(write_path, 'testfold_{}_pred_0.csv'.format(i)), index=False)
        else:
            X_test_pred_0.to_csv(os.path.join(write_path, 'testfold_{}_pred_0.csv'.format(i)), index=False)
        print(f'fold {i}: total instances: {ltest}, misclassified: {misclassified}, screened instances: {len(X_test_pred_0)} ')

def evaluate_model(data_x, data_y):
    """
    :param data_x:
    :param data_y:
    :return:
    """
    k_fold = KFold(10, shuffle=True, random_state=1)

    predicted_targets = np.array([])
    actual_targets = np.array([])

    for train_ix, test_ix in k_fold.split(data_x):
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]

        # Fit the classifier
        classifier = svm.SVC().fit(train_x, train_y)

        # Predict the labels of the test set samples
        predicted_labels = classifier.predict(test_x)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)

    return predicted_targets, actual_targets
