
#from dtreeviz.trees import dtreeviz
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
#import graphviz
from sklearn.tree import export_graphviz

class BuildTrees (object):
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model
        self.featurenames = ['Age', 'Experience', 'Income', 'Family', 'Education', 'Personal Loan', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
        self.classnames = ['0', '1']

    def plotTree_RF (self, tree_number):
        self.model.fit(self.X, self.y)
        #rf.estimators_[id] #to acces the individual tree
        fig = plt.figure(figsize=(15, 10))
        plot_tree(self.model.estimators_[tree_number],
          feature_names=self.featurenames,
          class_names=self.classnames,
          filled=True, impurity=True,
          rounded=True)
        #plt.show()
        #fig.savefig('rf_decisiontree'+'tree_number'+'.png')
    def plotTree_Textual_RF(self, tree_number):
        self.model.fit(self.X, self.y)
        #textual rule tree
        featurenames1 = ['Age', 'Experience', 'Income', 'Family', 'Education', 'Mortgage', 'Personal Loan', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
        print(export_text(self.model.estimators_[tree_number],
                  spacing=3, decimals=3,
                  feature_names=featurenames1))
