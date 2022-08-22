import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import sklearn.neural_network as sknet


def process_data_get_Xtest():
    bank = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\bank_updated_v1\\data_split_2test\\From Xtest_class0_2test_instances_bank.csv')
    adult = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\adult_results\\data_split_2test\\From Xtest_class0_2test_instances_adult.csv')
    credit = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\credit_results\\data_split_2test\\From Xtest_class0_2test_instances_credit.csv')
    appen = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\appendicitis\\train-test\\From Xtest_2_test_instances.csv')
    bupa = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\bupa\\train-test\\From Xtest_2_test_instances.csv')
    heart = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\heart\\train-test\\From Xtest_2_test_instances.csv')
    magic = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\magic\\train-test\\From Xtest_2_test_instances.csv')
    mammo = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\mammo\\train-test\\From Xtest_2_test_instances.csv')
    movie = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\movie\\train-test\\From Xtest_2_test_instances.csv')
    saheart = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\saheart\\train-test\\From Xtest_2_test_instances.csv')
    spotify = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\spotify\\train-test\\From Xtest_2_test_instances.csv')
    titanic = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\titanic\\train-test\\From Xtest_2_test_instances.csv')
    wdbc = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\wdbc\\train-test\\From Xtest_2_test_instances.csv')
    wine = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\wine-red\\train-test\\From Xtest_2_test_instances.csv')
    wiscon = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\wisconsin\\train-test\\From Xtest_2_test_instances.csv')
    return bank, credit, adult, appen, bupa, heart, magic, mammo, movie, saheart, spotify, titanic, wdbc, wine, wiscon

# need to open one by one dataset to get respective Xtest, a customized loop can be applied
def process_data_get_Xtrain():
    #Xtrain_bank = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\bank_updated_v1\\data_split_2test\\Xtrain_df_bank.csv')
    Xtrain_credit = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\credit_results\\data_split_2test\\Xtrain_df_credit.csv')
    #Xtrain_adult = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\adult_results\\data_split_2test\\Xtrain_df_adult.csv')
    #Xtrain_append = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\appendicitis\\train-test\\Xtrain_df.csv')
    #Xtrain_bupa = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\bupa\\train-test\\Xtrain_df.csv')
    #Xtrain_heart = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\heart\\train-test\\Xtrain_df.csv')
    #Xtrain_movie = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\movie\\train-test\\Xtrain_df.csv')
    #Xtrain_mammo = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\mammo\\train-test\\Xtrain_df.csv')
    #Xtrain_saheart = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\saheart\\train-test\\Xtrain_df.csv')
    #Xtrain_spotify = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\spotify\\train-test\\Xtrain_df.csv')
    #Xtrain_titanic = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\titanic\\train-test\\Xtrain_df.csv')
    #Xtrain_wdbc = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\wdbc\\train-test\\Xtrain_df.csv')
    #Xtrain_wine = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\wine-red\\train-test\\Xtrain_df.csv')
    Xtrain_wisconsin = pd.read_csv('C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\results_v2\\wisconsin\\train-test\\Xtrain_df.csv')
    return Xtrain_wisconsin

# logistic regression
def get_model():
    path = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE4Github\\'
    lg_bank = pickle.load(open(path+'\\classifiers\\lg_model_bank.sav', 'rb'))
    lg_credit = pickle.load(open(path+'\\classifiers\\lg_model_credit.sav', 'rb'))
    lg_adult = pickle.load(open(path+'\\classifiers\\lg_model_adult.sav', 'rb'))
    lg_append = pickle.load(open(path+'\\classifiers\\lg_model_appendicitis.sav', 'rb'))
    lg_bupa = pickle.load(open(path+'\\classifiers\\lg_model_bupa.sav', 'rb'))
    lg_heart = pickle.load(open(path+'\\classifiers\\lg_model_heart.sav', 'rb'))
    #lg_magic = pickle.load(open(path+'\\classifiers\\lg_model_magic.sav', 'rb'))
    lg_movie = pickle.load(open(path+'\\classifiers\\lg_model_movie.sav', 'rb'))
    lg_mammo = pickle.load(open(path+'\\classifiers\\lg_model_mammo.sav', 'rb'))
    lg_saheart = pickle.load(open(path+'\\classifiers\\lg_model_saheart.sav', 'rb'))
    lg_spotify = pickle.load(open(path+'\\classifiers\\lg_model_spotify.sav', 'rb'))
    lg_titanic = pickle.load(open(path+'\\classifiers\\lg_model_titanic.sav', 'rb'))
    lg_wdbc = pickle.load(open(path+'\\classifiers\\lg_model_wdbc.sav', 'rb'))
    lg_wine = pickle.load(open(path+'\\classifiers\\lg_model_wine-red.sav', 'rb'))
    lg_wiscon = pickle.load(open(path+'\\classifiers\\lg_model_wisconsin.sav', 'rb'))
    return lg_bank, lg_credit, lg_adult, lg_append, lg_bupa, lg_heart, lg_movie, lg_mammo, lg_saheart, lg_spotify, lg_titanic, lg_wdbc, lg_wine, lg_wiscon

def get_model_mlp():
    path = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE4Github\\'
    mlp_bank = pickle.load(open(path+'\\classifiers\\mlp_model_bank.sav', 'rb'))
    mlp_credit = pickle.load(open(path+'\\classifiers\\mlp_model_credit.pkl', 'rb'))
    mlp_adult = pickle.load(open(path+'\\classifiers\\mlp_model_adult.sav', 'rb'))
    mlp_append = pickle.load(open(path+'\\classifiers\\mlp_model_appendicitis.sav', 'rb'))
    mlp_bupa = pickle.load(open(path+'\\classifiers\\mlp_model_bupa.sav', 'rb'))
    mlp_heart = pickle.load(open(path+'\\classifiers\\mlp_model_heart.sav', 'rb'))
    # lg_magic = pickle.load(open(path+'\\classifiers\\lg_model_magic.sav', 'rb'))
    mlp_movie = pickle.load(open(path+'\\classifiers\\mlp_model_movie.sav', 'rb'))
    mlp_mammo = pickle.load(open(path+'\\classifiers\\mlp_model_mammo.sav', 'rb'))
    mlp_saheart = pickle.load(open(path+'\\classifiers\\mlp_model_saheart.sav', 'rb'))
    mlp_spotify = pickle.load(open(path+'\\classifiers\\mlp_model_spotify.sav', 'rb'))
    mlp_titanic = pickle.load(open(path+'\\classifiers\\mlp_model_titanic.sav', 'rb'))
    mlp_wdbc = pickle.load(open(path+'\\classifiers\\mlp_model_wdbc.sav', 'rb'))
    mlp_wine = pickle.load(open(path+'\\classifiers\\mlp_model_wine-red.sav', 'rb'))
    mlp_wiscon = pickle.load(open(path+'\\classifiers\\mlp_model_wisconsin.sav', 'rb'))
    return mlp_bank, mlp_credit, mlp_adult,  mlp_append, mlp_bupa, mlp_heart, mlp_movie, mlp_mammo, mlp_saheart, mlp_spotify, mlp_titanic, mlp_wdbc, mlp_wine, mlp_wiscon

    def classify_dataset_getModel(self, dataset_df, data_name=''):
        if data_name == 'bank':
            dataset_df.reset_index(drop=True, inplace=True)
            del dataset_df['Age']
            X = dataset_df.loc[:, dataset_df.columns != 'Personal Loan']
            y = dataset_df['Personal Loan']
        elif data_name == 'credit':
            dataset_df.reset_index(drop=True, inplace=True)
            X = dataset_df.loc[:, dataset_df.columns != 'Personal Loan']
            y = dataset_df['Personal Loan']
        elif data_name == 'cytometry':
            dataset_df.reset_index(drop=True, inplace=True)
            X = dataset_df.loc[:, dataset_df.columns != 'Class_Label']
            y = dataset_df['Class_Label']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)  # Split the data
        #dt = DecisionTreeClassifier()
        #dt = dt.fit(Xtrain, ytrain)
        #print("Accuracy DT:", metrics.accuracy_score(ytest, y_pred))
        #svm_model = svm.SVC()
        #svm_model.fit(Xtrain, ytrain)
        #svm_test = svm_model.score(Xtest, ytest)
        rf_mod = RandomForestClassifier(n_estimators=100,
                                        max_depth=3,
                                        max_features='auto',
                                        min_samples_leaf=4,
                                        bootstrap=True,
                                        n_jobs=-1,
                                        random_state=0)
        rf_mod.fit(Xtrain, ytrain)
        #rf_train = rf_mod.score(Xtrain, ytrain)
        #rf_cv = cross_val_score(rf_mod, Xtrain, ytrain, cv=5).mean()
        rf_test = rf_mod.score(Xtest, ytest)
        # print('Evaluation of the Random Forest performance\n')
        #print(f'Training score: {rf_train.round(4)}')
        # print(f'Cross validation score: {rf_cv.round(4)}')
        #dt_score = dt.score(Xtest,ytest)
        print(f'Test score RF: {rf_test.round(4)}')
        #print(f'Test score SVM: {svm_test.round(4)}')
        #print(f'Test score DT: {dt_score.round(4)}')
        return rf_mod,

