# -*- coding: utf-8 -*-
"""
Created on Tuesday 22 FEB 2022

@author: Muhammad Suffian
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from simplenlg.framework import *
from simplenlg.lexicon import *
from simplenlg.realiser.english import *
from simplenlg.phrasespec import *
from simplenlg.features import *

"""
 This is the module that could be utilized to take the user feedback in the form of user preferences
 Later these preferences would be taken from the user with an interface.

 The model prediction for the candidate couterfactuals should be done at the same loop and there should
 a dataframe to append the found counterfactuals, and return it by stopping the process.
"""

class UFCE():
    def __init__(self):
        #self.dataset = dataset
        #self.selected_features = user_selected_features
        #self.intervals = user_preferences
        #self.features = ['age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']
        """
        self.features = {'age': 'age',
                         'Experience': 'experience',
                         'Income': 'income',
                         'Family': 'family',
                         'CCAvg': 'CCAvg',
                         'Education': 'education',
                         'Mortgage': 'mortgage',
                         'SecuritiesAccount': 'securitiesAccount',
                         'CDAccount': "CDAccount",
                         'Online': 'online',
                         'CreditCard': 'credit card'
                         }
        """
        self.min_max_values = {'age': {'min': 21.0, 'max': 67.0},
                               'Experience': {'min': -3.0, 'max': 43.0},
                               'Income': {'min': 8.0, 'max': 224.0},
                               'Family': {'min': 1.0, 'max': 4.0},
                               'CCAvg': {'min': 0.0, 'max': 10.0},
                               'Education': {'min': 1.0, 'max': 4.0},
                               'Mortgage': {'min': 0.0, 'max': 635.0},
                               'SecuritiesAccount': {'min': 1.0, 'max': 2.0},
                               'CDAccount': {'min': 1.0, 'max': 2.0},
                               'Online': {'min': 1.0, 'max': 2.0},
                               'CreditCard': {'min': 1.0, 'max': 2.0}
                               }
        self.min_max_values_adult = {'workclass': {'min': 0, 'max': 8},
                               'fnlwgt': {'min': 12285, 'max': 1484705},
                               'education': {'min': 0, 'max': 15},
                               'education-num': {'min': 1, 'max': 16},
                               'marital-status': {'min': 0, 'max': 6},
                               'occupation': {'min': 0, 'max': 14},
                               'relationship': {'min': 0, 'max': 5},
                               'hours-per-week': {'min': 1, 'max': 99},
                               'native-country': {'min': 0, 'max': 41},
                               'net_capital': {'min': -4356, 'max': 99999},
                               'per-hour-income': {'min': -37, 'max': 218}
                               }
        self.min_max_values_compass = {'Number_of_Priors': {'min': 0, 'max': 38},
                                     'score_factor': {'min': 0, 'max': 1},
                                     'Age_Above_FourtyFive': {'min': 0, 'max': 1},
                                     'Age_Below_TwentyFive': {'min': 0, 'max': 1},
                                     'Misdemeanor': {'min': 0, 'max': 1}
                                     }
        self.min_max_values_diabetes = {'Pregnancies': {'min': 0.0, 'max': 17.0},
                               'Glucose': {'min': 0.0, 'max': 199.0},
                               'BloodPressure': {'min': 0.0, 'max': 122.0},
                               'SkinThickness': {'min': 0.0, 'max': 99.0},
                               'Insulin': {'min': 0.0, 'max': 846.0},
                               'BMI': {'min': 0.0, 'max': 67.0},
                               'DiabetesePedigreeFunction': {'min': 0.0, 'max': 2.42}
                               }
        self.min_max_values_wine = {'fixed acidity': {'min': 4.60, 'max': 15.90,},
                                     'volatile acidity': {'min': 0.23, 'max': 1.58},
                                     'citric acid': {'min': 0.0, 'max': 1.0},
                                     'residual sugar': {'min': 0.90, 'max': 15.50},
                                     'chlorides': {'min': 0.012, 'max': 0.61},
                                     'free sulfur dioxide': {'min': 1.0, 'max': 72.0},
                                     'total sulfur dioxide': {'min': 6.0, 'max': 289.0},
                                     'density': {'min': 0.99, 'max': 1.0},
                                     'pH': {'min': 2.74, 'max': 4.01},
                                     'sulphates': {'min': 0.33, 'max': 2.0},
                                     'alcohol': {'min': 8.40, 'max': 14.90}
                                     }

        self.min_max_values_credit = {'Married': {'min': 0.0, 'max': 1.0},
                                        'Single': {'min': 0.0, 'max': 1.0},
                                        'Age_lt_25': {'min': 0.0, 'max': 1.0},
                                        'Age_in_25_to_40': {'min': 0.0, 'max': 1.0},
                                        'Age_in_40_to_59': {'min': 0.0, 'max': 1.0},
                                        'Age_geq_60': {'min': 0.0, 'max': 1.0},
                                        'EducationLevel': {'min': 0.0, 'max': 3.0},
                                      'MaxBillAmountOverLast6Months': {'min': 0.0, 'max': 50810.0},
                                      'MaxPaymentAmountOverLast6Months': {'min': 0.0, 'max': 51430.0},
                                      'MonthsWithZeroBalanceOverLast6Months': {'min': 0.0, 'max': 6.0},
                                      'MonthsWithLowSpendingOverLast6Months': {'min': 0.0, 'max': 6.0},
                                      'MostRecentBillAmount': {'min': 0.0, 'max': 29450.0},
                                      'MostRecentPaymentAmount': {'min': 0.0, 'max': 26670.0},
                                      'TotalOverdueCounts': {'min': 0.0, 'max': 3.0},
                                      'TotalMonthsOverdue': {'min': 0.0, 'max': 36.0},
                                      'HistoryOfOverduePayments': {'min': 0.0, 'max': 1.0}
                                        }
        self.min_max_values_wisconsin = {'ClumpThickness': {'min': 1, 'max': 10, },
                                    'CellSize': {'min': 1, 'max': 10},
                                    'CellShape': {'min': 1, 'max': 10},
                                    'MarginalAdhesion': {'min': 1, 'max': 10},
                                    'EpithelialSize': {'min': 1, 'max': 10},
                                    'BareNuclei': {'min': 1, 'max': 10},
                                    'BlandChromatin': {'min': 1, 'max': 10},
                                    'NormalNucleoli': {'min': 1, 'max': 10},
                                    'Mitoses': {'min': 1, 'max': 10}
                                    }
        self.min_max_values_bupa = {'Mcv': {'min': 65, 'max': 103},
                                         'Alkphos': {'min': 23, 'max': 138},
                                         'Sgpt': {'min': 4, 'max': 155},
                                         'Sgot': {'min': 5, 'max': 82},
                                         'Gammagt': {'min': 5, 'max': 297},
                                         'Drinks': {'min': 0, 'max': 20}
                                         }
        self.min_max_values_appendicitis = {'At1': {'min': 0.0, 'max': 1.0},
                                            'At2': {'min': 0.0, 'max': 1.0},
                                            'At3': {'min': 0.0, 'max': 1.0},
                                            'At4': {'min': 0.0, 'max': 1.0},
                                            'At5': {'min': 0.0, 'max': 1.0},
                                            'At6': {'min': 0.0, 'max': 1.0},
                                            'At7': {'min': 0.0, 'max': 1.0}
                                    }
        self.min_max_values_saheart = {'Sbp': {'min': 101, 'max': 218},
                                            'Tobacco': {'min': 0.0, 'max': 31.2},
                                            'Ldl': {'min': 0.98, 'max': 15.33},
                                            'Adiposity': {'min': 6.74, 'max': 42.49},
                                            'Famhist': {'min': 0, 'max': 1},
                                            'Typea': {'min': 13, 'max': 78},
                                            'Obesity': {'min': 14.7, 'max': 46.58},
                                            'Alcohol': {'min': 0.0, 'max': 147.19},
                                            'Age': {'min': 15, 'max': 64}
                                            }

        self.min_max_values_spotify = {'acousticness': {'min': 0.0, 'max': 0.995},
                                       'danceability': {'min': 0.122, 'max': 0.98},
                                       'duration_ms': {'min': 16.0, 'max': 1004.0},
                                       'energy': {'min': 0.014, 'max': 0.99},
                                       'instrumentalness': {'min': 0.0, 'max': 0.97},
                                       'key': {'min': 0.0, 'max': 11.0},
                                       'liveness': {'min': 0.018, 'max': 0.96},
                                       'loudness': {'min': -33.09, 'max': -0.30},
                                       'mode': {'min': 0.0, 'max': 1.0},
                                       'speechiness': {'min': 0.023, 'max': 0.81},
                                       'tempo': {'min': 47.85, 'max': 219.31},
                                       'time_signature': {'min': 1.0, 'max': 5.0},
                                       'valence': {'min': 0.034, 'max': 0.99}
                                       }

        self.min_max_values_movie = {'Marketing expense': {'min': 20.12, 'max': 1799.52},
                                       'Production expense': {'min': 55.92, 'max': 110.48},
                                       'Multiplex coverage': {'min': 0.12, 'max': 0.61},
                                       'Budget': {'min': 19781.35, 'max': 48772.90},
                                       'Movie_length': {'min': 76.40, 'max': 173.50},
                                       'Lead_ Actor_Rating': {'min': 3.8, 'max': 9.4},
                                       'Lead_Actress_rating': {'min': 4.03, 'max':9.5 },
                                       'Director_rating': {'min': 3.8, 'max': 9.4},
                                       'Producer_rating': {'min': 4.03, 'max': 9.6},
                                       'Critic_rating': {'min': 6.6, 'max': 9.4},
                                       'Trailer_views': {'min': 212912, 'max': 567784},
                                       '3D_available': {'min': 0.0, 'max': 1.0},
                                       'Time_taken': {'min': 0.0, 'max': 217.52},
                                     'Twitter_hastags': {'min': 201.15, 'max': 2022.4},
                                     'Genre': {'min': 1.0, 'max': 4.0},
                                     'Avg_age_actors': {'min': 3.0, 'max': 60.0},
                                     'Num_multiplex': {'min': 333.0, 'max': 868.0},
                                     'Collection': {'min': 10000.0, 'max': 100000.0}
                                       }
        self.min_max_values_heart = {'Age': {'min': 29, 'max': 77},
                                       'Sex': {'min': 0, 'max': 1},
                                       'ChestPainType': {'min': 1, 'max': 4},
                                       'RestBloodPressure': {'min': 94, 'max': 200},
                                       'SerumCholestoral': {'min': 126, 'max': 564},
                                       'FastingBloodSugar': {'min': 0, 'max': 1},
                                       'ResElectrocardiographic': {'min': 0, 'max': 2},
                                       'MaxHeartRate': {'min': 71, 'max': 202},
                                       'ExerciseInduced': {'min': 0, 'max': 1},
                                     'Oldpeak': {'min': 0.0, 'max': 62.0},
                                     'Slope': {'min': 1, 'max': 3},
                                     'MajorVessels': {'min': 0, 'max': 3},
                                     'Thal': {'min': 3, 'max': 7}
                                    }
        self.min_max_values_wdbc = {'Radius1': {'min': 6.98, 'max': 28.11},
                                     'Texture1': {'min': 9.71, 'max': 39.28},
                                     'Perimeter1': {'min': 43.79, 'max': 188.5},
                                     'Area1': {'min': 143.0, 'max': 2501.0},
                                     'Smoothness1': {'min': 0.053, 'max': 0.163},
                                     'Compactness1': {'min': 0.019, 'max': 0.345},
                                     'Concavity1': {'min': 0.0, 'max': 0.42},
                                     'Concave_points1': {'min': 0.0, 'max': 0.20},
                                     'Symmetry1': {'min': 0.106, 'max': 0.304},
                                     'Fractal_dimension1': {'min': 0.05, 'max': 0.097},
                                     'Radius2': {'min': 0.112, 'max': 2.873},
                                     'Texture2': {'min': 0.36, 'max': 4.88},
                                    'Perimeter2': {'min': 0.75, 'max': 21.98},
                                     'Area2': {'min': 6.8, 'max': 542.2},
                                    'Smoothness2': {'min': 0.0020, 'max': 0.031},
                                    'Compactness2': {'min': 0.0020, 'max': 0.032},
                                    'Concavity2': {'min': 0.0, 'max': 0.396},
                                    'Concave_points2': {'min': 0.0, 'max': 0.053},
                                    'Symmetry2': {'min': 0.0080, 'max': 0.079},
                                    'Fractal_dimension2': {'min': 0.0010, 'max': 0.03},
                                    'Radius3': {'min': 7.93, 'max': 36.04},
                                    'Texture3': {'min': 12.02, 'max': 49.54},
                                    'Perimeter3': {'min': 50.4, 'max': 251.2},
                                    'Area3': {'min': 185.2, 'max': 4254.0},
                                    'Smoothness3': {'min': 0.071, 'max': 0.223},
                                    'Compactness3': {'min': 0.027, 'max': 1.058},
                                    'Concavity3': {'min': 0.0, 'max': 1.252},
                                    'Concave_points3': {'min': 0.0, 'max': 0.291},
                                    'Symmetry3': {'min': 0.156, 'max': 0.664},
                                    'Fractal_dimension3': {'min': 0.055, 'max': 0.208},
                                     }
        self.min_max_values_magic = {'FLength': {'min': 4.28, 'max': 334.17},
                                     'FWidth': {'min': 0.0, 'max': 256.38},
                                     'FSize': {'min': 1.94, 'max': 5.32},
                                     'FConc': {'min': 0.01, 'max': 0.89},
                                     'FConc1': {'min': 0.00030, 'max': 0.67},
                                     'FAsym': {'min': -457.910, 'max': 575.24},
                                     'FM3Long': {'min': -331.78, 'max': 238.32},
                                     'FM3Trans': {'min': -205.98, 'max': 179.85},
                                     'FAlpha': {'min': 0.0, 'max': 90.0},
                                     'FDist': {'min': 1.28, 'max': 495.56}
                                     }
        self.min_max_values_titanic = {'Class': {'min': 1.0, 'max': 4.0},
                                     'Age': {'min': 1.0, 'max': 2.0},
                                     'Sex': {'min': 0.0, 'max': 1.0}
                                     }
        self.min_max_values_mammo = {'BI-RADS': {'min': 0, 'max': 6},
                                        'Age': {'min': 18, 'max': 96},
                                       'Shape': {'min': 1, 'max': 4},
                                       'Margin': {'min': 1, 'max': 5},
                                     'Density': {'min': 1, 'max': 4}
                                       }

        self.output_classes = {'1': 'Personal_loan_denied',
                               '2': 'Personal_loan_granted',
                               }

        self.candidate_counterfactuals_set = dict()
        self.counterfactuals = dict()

    def identify_test_instance_bucket(self, test_instance, features, protected_features ):
        """
        :param test_instance: The identification of bucket for test instance
        :param intervals: The interval belongs to LOW, MEDiUM and HIGH, to which test_instance will belong
        :return: a bucket or list of start and end values for a specific feature.
        """
        buckets_dict  = {'age':[],
                        'Experience': [],
                        'Income': [],
                        'Family': [],
                        'CCAvg': [],
                        'Education': [],
                        'Mortgage': [],
                        'SecuritiesAccount': [],
                        'CDAccount': [],
                        'Online': [],
                        'CreditCard': []
                        }
        for feature in features:
            if feature not in protected_features:
                if feature == 'age':
                    if (36.33 > float(test_instance[feature].values) >= 21.05):
                        buckets_dict[feature].append('age-Young')
                    elif (51.67 > float(test_instance[feature].values) >= 36.33):
                        buckets_dict[feature].append('age-Middle')
                    else: buckets_dict[feature].append('age-Old')
                elif feature == 'Experience':
                    if (12.33 > float(test_instance[feature].values) >= -3.0):
                        buckets_dict[feature].append('Exp-Low')
                    elif (27.67 > float(test_instance[feature].values) >= 12.33):
                        buckets_dict[feature].append('Exp-Medium')
                    else: buckets_dict[feature].append('Exp-High')
                elif feature == 'Income':
                    if (80.0 > float(test_instance[feature].values) >= 8.0):
                        buckets_dict[feature].append('Income-Low')
                    elif (152.0 > float(test_instance[feature].values) >= 80.0):
                        buckets_dict[feature].append('Income-Medium')
                    else: buckets_dict[feature].append('Income-High')
                elif feature == 'Family':
                    if (2.0 > float(test_instance[feature].values) >= 1.0):
                        buckets_dict[feature].append('Family-small')
                    elif (3.0 > float(test_instance[feature].values) >= 2.0):
                        buckets_dict[feature].append('Family-Medium')
                    else: buckets_dict[feature].append('Family-Large')
                elif feature == 'CCAvg':
                    if (3.33 > float(test_instance[feature].values) >= 0.0):
                        buckets_dict[feature].append('CCAvg-Low')
                    elif (6.67 > float(test_instance[feature].values) >= 3.33):
                        buckets_dict[feature].append('CCAvg-Medium')
                    else: buckets_dict[feature].append('CCAvg-High')
                elif feature == 'Education':
                    if (1.5 > float(test_instance[feature].values) >= 1.0):
                        buckets_dict[feature].append('Education-undergraduate')
                    elif (3.5 > float(test_instance[feature].values) >= 2.5):
                        buckets_dict[feature].append('Education-graduate')
                    elif (4.0 > float(test_instance[feature].values) >= 3.5):
                        buckets_dict[feature].append('Education-professional')
                    else: buckets_dict[feature].append('Education-Unknown')
                elif feature == 'Mortgage':
                    if (211.67 > float(test_instance[feature].values) >= 0.0):
                        buckets_dict[feature].append('Mortgage-Low')
                    elif (433.33 > float(test_instance[feature].values) >= 211.67):
                        buckets_dict[feature].append('Mortgage-Medium')
                    else: buckets_dict[feature].append('Mortgage-High')
                elif feature == 'SecuritiesAccount':
                    if (float(test_instance[feature].values) == 0.0):
                        buckets_dict[feature].append('SecuritiesAccount-No_sec')
                    else: buckets_dict[feature].append('SecuritiesAccount-Yes_sec')
                elif feature == 'CDAccount':
                    if (float(test_instance[feature].values) == 0.0):
                        buckets_dict[feature].append('CDAccount-No_certificate')
                    else: buckets_dict[feature].append('CDAccount-Yes_certificate')
                elif feature == 'Online':
                    if (float(test_instance[feature].values) == 0.0):
                        buckets_dict[feature].append('Online-No_internet')
                    else: buckets_dict[feature].append('Online-Yes_internet')
                elif feature == 'CreditCard':
                    if (float(test_instance[feature].values) == 0.0):
                        buckets_dict[feature].append('CreditCard-No_card')
                    else: buckets_dict[feature].append('CreditCard-Yes_card')
            else:
                pass
        return buckets_dict

#categorical data handler function
    def categorical_handler(self, test_instance, user_cat_feature_list):
        for feature in user_cat_feature_list:
            if float(test_instance.loc[:, feature].values) != 1:
                test_instance.loc[:, feature] = 1.0
        return test_instance


#testng the new feature in this method
    def one_feature_synthetic_data_testing(self, test_instance, u_cat_f_list, user_term_intervals, protected_features,
                                   perturbing_rates, model, desired_outcome, k):
        #print("One-feature function here")
        # path1 = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\bank_updated\\'
        # bucket_ranges = self.identify_test_instance_bucket(test_instance, protected_features)
        one_feature_dataframe = pd.DataFrame()
        cfdf = pd.DataFrame()
        tstinst = pd.DataFrame()
        tempdf = pd.DataFrame()
        tempdf = test_instance.copy()
        found = 0
        f = 'one_feature_perturbed_at_time'
        for feature in user_term_intervals:
            one_feature_data = pd.DataFrame()
            interval_term_range = user_term_intervals[feature]
            #print(interval_term_range, feature)
            if len(interval_term_range) != 0:
                start = interval_term_range[0]
                end = interval_term_range[1]
                number_of_iterations = (end - start) / perturbing_rates[feature]
                # print("feature, start, end, iterations:", feature, start, end, number_of_iterations)
                tempdf.loc[:, feature] = start
                for iter in range(int(number_of_iterations)):
                    flag = 0  # toclose
                    if found == k:
                        break
                    else:
                        tempdf.loc[:, feature] = float(tempdf.loc[:, feature].values + perturbing_rates[feature])
                        #for uf in u_cat_f_list:
                            #print(uf, type(tempdf))
                            #uf = str(uf)
                            #if float(tempdf.loc[:, uf].values) != 1.0:
                                #tempdf.loc[:, uf] = 1.0
                        pred = model.predict(tempdf)
                        if pred == desired_outcome:
                            cfdf = pd.concat([cfdf, tempdf], ignore_index=True, axis=0)
                            found = found + 1
                            #print("found CFs for:", feature, iter, found)
                            #break
                if found == k:
                    break        #count += 1

        #if foun1 == 1:
            #break            #count += 1
                            #print("found")
                            #if count > 5:
                            #    print("5 cfs here")

                    #one_feature_data = pd.concat([one_feature_data, tempdf], ignore_index=True, axis=0)
                #one_feature_dataframe = pd.concat([one_feature_dataframe, one_feature_data], ignore_index=True, axis=0)
        #cfdf.to_csv(path + '' + f + '' + '.csv')
        #print("after break:", count)

        return cfdf

    # def one_feature_synthetic_data(self, test_instance, user_term_intervals, protected_features, path, perturbing_rates):
    #
    #     #path1 = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\bank_updated\\'
    #     #bucket_ranges = self.identify_test_instance_bucket(test_instance, protected_features)
    #     one_feature_dataframe = pd.DataFrame()
    #     f = 'one_feature_perturbed_at_time'
    #     for feature in user_term_intervals:
    #         one_feature_data = pd.DataFrame()
    #         tempdf = pd.DataFrame()
    #         tempdf = test_instance.copy()
    #         #interval_term = bucket_ranges[feature]
    #         interval_term_range = user_term_intervals[feature]
    #         if len(interval_term_range) != 0:
    #             start = interval_term_range[0]
    #             end = interval_term_range[1]
    #             number_of_iterations = (end - start) / perturbing_rates[feature]
    #             #print("feature, start, end, iterations:", feature, start, end, number_of_iterations)
    #             tempdf.loc[:, feature] = start
    #             for iter in range (int (number_of_iterations)):
    #                 tempdf.loc[:, feature] = float(tempdf.loc[:, feature].values + perturbing_rates[feature])
    #                 one_feature_data = pd.concat([one_feature_data, tempdf], ignore_index=True, axis=0)
    #             one_feature_dataframe = pd.concat([one_feature_dataframe, one_feature_data], ignore_index=True, axis=0)
    #     one_feature_dataframe.to_csv(path+''+f+''+'.csv')
    #
    #     return one_feature_dataframe

    def get_highly_correlated(self, df, features, threshold=0.5):
        #print(features)
        corr_df = df[features].corr()  # get correlations
        correlated_features = np.where(np.abs(corr_df) > threshold)
        correlated_features = [(corr_df.iloc[x, y], x, y) for x, y in zip(*correlated_features) if x != y and x < y]  # avoid duplication
        s_corr_list = sorted(correlated_features, key=lambda x: -abs(x[0]))  # sort by correlation value
        corr_dict = dict()
        if s_corr_list == []:
            print("There are no highly correlated features with correlation:", threshold)
        else:
            for v, i, j in s_corr_list:
                cols = df[features].columns
                corr_dict[corr_df.index[i]] = corr_df.columns[j]
                #print("%s and %s = %.3f" % (corr_df.index[i], corr_df.columns[j], v))

        keys_list = corr_dict.keys()
        feature_list = []
        features_to_use = []
        for key in keys_list:
            feature_list.append(key)
            feature_list.append(corr_dict[key])
        features_to_use.append(feature_list[0])
        features_to_use.append(feature_list[1])
        #features_to_use.append(feature_list[2])
        #print("suggested-corr-features, feature_list:", features_to_use, feature_list)
        return corr_dict, features_to_use

    def two_feature_synthetic_data(self, df, test_instance, user_corr_features, u_cat_f_list, user_term_intervals, features, perturbing_rates, model, desired_outcome, k):
        #path2 = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\bank_updated\\'
        #bucket_ranges = self.identify_test_instance_bucket(test_instance, protected_features)
        count = 0
        cfdf = pd.DataFrame()
        two_feature_dataframe = pd.DataFrame()
        ff = 'two_feature_perturbed_at_time'
        #here finding the highly correlated features in case user don't specify any correlation
        if user_term_intervals != [] and len(features) >= 2:
            corr_features_dict, feature_to_use_list = self.get_highly_correlated(df, features)
        else:
            feature_to_use_list = user_corr_features
        f1 = str(feature_to_use_list[0])
        f2 = str(feature_to_use_list[1])
        #print('f1, f2, f3',f1,f2,f3)
        two_feature_data = pd.DataFrame()
        tempdf1 = pd.DataFrame()
        tempdf1 = test_instance.copy()
        tempdf2 = pd.DataFrame()
        tempdf2 = test_instance.copy()
        #calculating iteration for f1
        #print("here",user_term_intervals, f1)
        interval_term_range1 = user_term_intervals[f1]
        start1 = interval_term_range1[0]
        end1 = interval_term_range1[1]
        number_of_iterations1 = (end1 - start1) / perturbing_rates[f1]
        # calculating iteration for f2
        #print(f2, user_term_intervals)
        interval_term_range2 = user_term_intervals[f2]
        start2 = interval_term_range2[0]
        end2 = interval_term_range2[1]
        number_of_iterations2 = (end2 - start2) / perturbing_rates[f2]
        #print("two feature: f1-iterations, f2-iterations", number_of_iterations1, number_of_iterations2)
        tempdf1.loc[:, f1] = start1
        flag1 = 0
        for iter in range (int (number_of_iterations1)):
            tempdf1.loc[:, f1] = tempdf1.loc[:, f1].values + perturbing_rates[f1]
            #two_feature_data = pd.concat([two_feature_data, tempdf], ignore_index=True, axis=0)
            #two_feature_dataframe = pd.concat([two_feature_dataframe, two_feature_data], ignore_index=True, axis=0)
            tempdf1.loc[:, f2] = start2

            for iter2 in range (int (number_of_iterations2)):
                flag = 0  # toclose
                tempdf1.loc[:, f2] = tempdf1.loc[:, f2].values + perturbing_rates[f2]
                #for r in u_cat_f_list: #to reverse the actual values of the cat-features
                #    tempdf1.loc[:,r] = test_instance.loc[:, r].values
                #for uf in u_cat_f_list: #cat-features perturbing same time
                #    if float(tempdf1.loc[:, uf].values) != 1.0:
                #        tempdf1.loc[:, uf] = 1.0
                #print(tempdf1.columns)
                pred = model.predict(tempdf1)
                if pred == desired_outcome:  #02 for benign, we try to convert 4 malignant into 2
                    cfdf = pd.concat([cfdf, tempdf1], ignore_index=True, axis=0)
                    flag1 += 1
                    #print("found")
                    break
            if flag1 == k:
                break    #count += 1
            #break        #print("found in two-feature")
                        #if count > 5:
                        #    break

                #two_feature_data = pd.concat([two_feature_data, tempdf1], ignore_index=True, axis=0)
            #print(cfdf)
            #two_feature_dataframe = pd.concat([two_feature_dataframe, two_feature_data], ignore_index=True, axis=0)
            #two_feature_dataframe = pd.concat([two_feature_dataframe, tempdf1], ignore_index=True, axis=0)
        #two_feature_dataframe = two_feature_dataframe.transform(np.sort)
        #two_feature_dataframe = two_feature_dataframe.drop_duplicates(keep='first')
        #two_feature_dataframe.to_csv(path+''+f+''+'.csv')
        #cfdf.to_csv(path+''+ff+''+'.csv')
        #print("two-feature_cfs:", count)
        return cfdf

    def three_feature_dynamic_synthetic_data(self, df, test_instance, user_corr_features, user_term_intervals, u_cat_f_list, features, perturbing_rates, model, desired_outcome, k):
        #path4 = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\bank_updated\\'
        #bucket_ranges = self.identify_test_instance_bucket(test_instance, protected_features)
        count = 0
        cfdf = pd.DataFrame()
        three_feature_dataframe = pd.DataFrame()
        three_feature_dataframe = test_instance.copy()
        three_f_df_2_output = pd.DataFrame()
        three_f_df_2_output = test_instance.copy()
        fff = 'three_feature_dynamic_perturbed_at_time'
        #print(self.get_highly_correlated(df, features))
        if user_corr_features == [] and len(features) >= 2:  #user_term_intervals replaced with user_corrfeatures
            corr_features_dict, feature_to_use_list = self.get_highly_correlated(df, features)
            #print("in system corr features")
        else:
            feature_to_use_list = user_corr_features
            #print("in the user corr-features")
        f1 = feature_to_use_list[0]
        f2 = feature_to_use_list[1]
        f3 = feature_to_use_list[2]
        one_feature_data = pd.DataFrame()
        two_feature_data = pd.DataFrame()
        three_feature_data = pd.DataFrame()
        tempdf1 = pd.DataFrame()
        tempdf1 = test_instance.copy()
        tempdf2 = pd.DataFrame()
        tempdf2 = test_instance.copy()
        tempdf3 = pd.DataFrame()
        tempdf3 = test_instance.copy()
        three_feature_dataframe = test_instance.copy()
        #calculating iterations
        interval_ranges1 = user_term_intervals[f1]
        interval_ranges2 = user_term_intervals[f2]
        interval_ranges3 = user_term_intervals[f3]
        start1 = interval_ranges1[0]
        end1 = interval_ranges1[1]
        number_of_iterations1 = (end1 - start1) / perturbing_rates[f1]
        if number_of_iterations1 == 0:
            number_of_iterations1 = 1
        # calculating iteration for f2
        start2 = interval_ranges2[0]
        end2 = interval_ranges2[1]
        number_of_iterations2 = (end2 - start2) / perturbing_rates[f2]
        if number_of_iterations2 == 0:
            number_of_iterations2 = 1
        start3 = interval_ranges3[0]
        end3 = interval_ranges3[1]
        number_of_iterations3 = (end3 - start3) / perturbing_rates[f3]
        if number_of_iterations3 == 0:
            number_of_iterations3 = 1
        x = 0
        y = 0
        z = 0
        flag = 0
        #print("three feature: f1-iterations, f2-iterations, f3-iterations", number_of_iterations1, number_of_iterations2, number_of_iterations3)
        tempdf1.loc[:, f1] = start1
        flag=0
        for x in range(int (number_of_iterations1)):
            #cfdf = pd.DataFrame()
            tempdf1.loc[:,f1] = float(tempdf1.loc[:,f1].values) + perturbing_rates[f1]
            #limit1 = end1
            #print("limit1",limit1)
            #if limit1 > end1:
            #    tempdf1 = test_instance.copy()
            tempdf2.loc[:, f2] = start2
            #flag = 0 #toclose
            for y in range(int(number_of_iterations2)):
                tempdf2.loc[:,f2] = float(tempdf2.loc[:, f2].values) + perturbing_rates[f2]
                #limit2 = end2
                #print("limit2", limit2)
                #if limit2 > end2:
                #    tempdf2 = test_instance.copy()
                tempdf3.loc[:, f3] = start3
                #flag = 0 #toclose
                for z in range(int(number_of_iterations3)):
                    tempdf3.loc[:, f3] = float(tempdf3.loc[:,f3].values) + perturbing_rates[f3]
                    #limit3 = end3
                    #print("limit3", limit3)
                    #if limit3 > end3:
                    #    tempdf3 = test_instance.copy()
                    three_feature_dataframe.loc[:, f1] = float(tempdf1.loc[:, f1].values)#topen
                    three_feature_dataframe.loc[:, f2] = float(tempdf2.loc[:, f2].values)#topen
                    three_feature_dataframe.loc[:, f3] = float(tempdf3.loc[:, f3].values)#topen
                    #for r in u_cat_f_list:  # to reverse the actual values of the cat-features
                    #    tempdf3.loc[:, r] = test_instance.loc[:, r].values
                    #for uf in u_cat_f_list:  # cat-features perturbing same time
                    #if float(tempdf3.loc[:, uf].values) != 1.0:
                    #    tempdf3.loc[:, uf] = 1.0
                    pred = model.predict(three_feature_dataframe)
                    if pred == desired_outcome:  #02 for benign, we try to convert 4 malignant into 2
                        cfdf = pd.concat([cfdf, three_feature_dataframe], ignore_index=True, axis=0)
                        flag += 1
                        break
                if flag==k:
                    break        #count += 1
            if flag==k:
                break
        #if flag==k:
        #    break            #print("found in three-feature")
            #break                #break
                    #print("df end point",float(three_feature_dataframe.loc[:, f1].values),float(three_feature_dataframe.loc[:, f2].values), float(three_feature_dataframe.loc[:, f3].values) )
                    #three_f_df_2_output = pd.concat([three_f_df_2_output, three_feature_dataframe], ignore_index=True, axis=0)
                    #three_feature_dataframe = pd.concat([three_feature_dataframe, tempdf2], ignore_index=True, axis=0)
                    #three_feature_dataframe = pd.concat([three_feature_dataframe, tempdf3], ignore_index=True, axis=0)
                    #print("tempdf1",tempdf1)
                    #z = float(z + perturbing_rates[f3])
                    #print("here in 3f")
                #three_feature_dataframe.loc[:, f1] = float(tempdf1.loc[:, f1].values)#toopen
                #three_feature_dataframe.loc[:, f2] = float(tempdf2.loc[:, f2].values)#toopen
                #three_feature_dataframe.loc[:, f3] = float(tempdf3.loc[:, f3].values)
                #three_f_df_2_output = pd.concat([three_f_df_2_output, three_feature_dataframe], ignore_index=True, axis=0)#topen
                #y = float(y + perturbing_rates[f2])
            #print("tempdf3", tempdf3)
            #three_feature_dataframe.loc[:, f1] = float(tempdf1.loc[:, f1].values)#topen
            #three_f_df_2_output = pd.concat([three_f_df_2_output, three_feature_dataframe], ignore_index=True,axis=0)topen
            #x = float(x + perturbing_rates[f1])
            #print("look here",three_feature_dataframe,three_f_df_2_output)
        #three_f_df_2_output = three_f_df_2_output.transform(np.sort)
        #three_f_df_2_output = three_f_df_2_output.drop_duplicates(keep='first') #topen
        #three_f_df_2_output.to_csv(path+''+f+''+'.csv')#topen
        #cfdf.to_csv(path+''+fff+''+'.csv')
        return cfdf


    def candidate_counterfactuals_df(self, df1, df2, df3, path):
        #path = 'C:\\Users\\laboratorio\\Documents\\Suffian PhD Work\\codes\\UFCE\\data\\'
        f = 'Final_merged_df_with_all_combinations'
        df_2_return = pd.DataFrame()
        df_2_return = pd.concat([df1, df2], ignore_index=True, axis=0)
        df_2_return = pd.concat([df_2_return, df3], ignore_index=True, axis=0)
        df_2_return = df_2_return.transform(np.sort)
        #df_2_return = df_2_return.drop_duplicates(keep='first')
        df_2_return.to_csv(path + '' + f + '' + '.csv')
        return df_2_return

    # considering user feedback for feature changes and aligning with intervals
    def user_feedback_processing(self, test_instance, user_feature_list=[], feature_flags={}, threshold_values={}, order_of_asymmetric={}):
        make_interval = dict()
        if len(user_feature_list)==len(feature_flags)==len(threshold_values):
            for feature in user_feature_list:  #using range-len only for compass to handle its no feature-in the df
                feature_flag = feature_flags[feature]
                if feature_flag == 'S': #threshold to +- (symmetric change)
                    threshold = threshold_values[feature]
                    #threshold (+) to add to make end of interval
                    end = test_instance[feature].values + threshold #removed .values values for compass
                    # threshold (-) to subtract to make start of interval
                    start = test_instance[feature].values - threshold #removed .values for compass
                    #if start < self.min_max_values_compass[feature]['min']: #new rule added for compass
                    #    start = test_instance[feature].values
                    make_interval[feature] = [start[0], end[0]]
                else: # 'A', asymmetric change
                    if order_of_asymmetric[feature] == 'I': #increasing order, add will make end of interval
                        threshold = threshold_values[feature]
                        #print(test_instance[feature].values)
                        #print(feature)
                        end = test_instance[feature].values + threshold #removed .values for the compass
                        start = test_instance[feature].values
                        #print("startend", start, end) # to check for compass
                        make_interval[feature] = [start[0], end[0]] #removing subscript start[0] and end[0] only for compass
                    else: # 'D', decreasing order, subtract will make start of inetrval
                        threshold = threshold_values[feature]
                        start = test_instance[feature].values - threshold #removed .values for compass
                        end = test_instance[feature].values #removed .values for compass
                        make_interval[feature] = [start[0], end[0]]
        return make_interval


    def classify_dataset_getModel(self, dataset_df, data_name=''):
        if data_name == 'spotify':
            dataset_df.reset_index(drop=True, inplace=True)
            X = dataset_df.loc[:, dataset_df.columns != 'like_dislike']
            y = dataset_df['like_dislike']
        elif data_name == 'bank':
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
        svm_model = svm.SVC()
        svm_model.fit(Xtrain, ytrain)
        svm_test = svm_model.score(Xtest, ytest)
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
        print(f'Test score SVM: {svm_test.round(4)}')
        #print(f'Test score DT: {dt_score.round(4)}')
        return svm_model, rf_mod,

    def get_model(self, df, path):
        #from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        import sklearn.neural_network as sknet
        import pickle
        from sklearn.model_selection import train_test_split, cross_val_score
        xtr = 'Xtrain_df'
        xts = 'Xtest_df'
        ytr = 'ytrain_df'
        yts = 'ytest_df'
        xtest_2test = 'From Xtest_2_test_instances'

        X = df.loc[:, df.columns != 'Severity']
        y = df['Severity']
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, stratify=y, shuffle=True, random_state=42)
        #X = df.loc[:, df.columns != 'Personal Loan']
        #y = df['Personal Loan']
        #X = df.loc[:, df.columns != ' class']
        #y = df[' class']
        #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,test_size=0.25, stratify=df[' class'], shuffle=True, random_state=42)  # Split the data

        #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)  # Split the data

        #feature_columns = ['Number_of_Priors', 'score_factor', 'Age_Above_FourtyFive', 'Age_Below_TwentyFive',
        #                   'Misdemeanor']
        #X = df[feature_columns]
        #y = df['Two_yr_Recidivism']
        # Create train and validation set

        #X = df.loc[:, df.columns != 'Outcome']
        #y = df['Outcome']
        #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, stratify=df['Outcome'], shuffle=True, random_state=42)

        #X = df.loc[:, df.columns != 'NoDefaultNextMonth']
        #y = df['NoDefaultNextMonth']
        #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, stratify=df['NoDefaultNextMonth'], shuffle=True, random_state=42)

        #Xtrain = pd.read_csv(path + '' + xtr + '' + '.csv')
        #ytrain = pd.read_csv(path + '' + ytr + '' + '.csv')
        #Xtest = pd.read_csv(path + '' + xts + '' + '.csv')
        #ytest = pd.read_csv(path + '' + yts + '' + '.csv')
        #print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
        #del Xtrain['Unnamed: 0']
        #print(Xtrain.head())
        #print(ytrain.columns)

        #del ytrain['Unnamed: 0']
        #del Xtest['Unnamed: 0']
        #del ytest['Unnamed: 0']
        #Xtrainm = Xtrain.values
        #ytrainm = ytrain.values
        #Xtestm = Xtest.values
        lg = LogisticRegression(max_iter = 1000)
        lg.fit(Xtrain, ytrain) #ytrain.iloc[:,1]
        print("Test Score Logistic Regression: {:.2%}".format(lg.score(Xtest, ytest))) #ytest.iloc[:,1]
        # model = RandomForestClassifier(n_estimators=100,
        #                               max_depth=3,
        #                               max_features='auto',
        #                               min_samples_leaf=4,
        #                               bootstrap=True,
        #                               n_jobs=-1,
        #                               random_state=0)
        #mlp = sknet.MLPClassifier(hidden_layer_sizes=(100,), random_state=42)
        #mlp.fit(Xtrain, ytrain) #iloc[:,1]
        #print("Test Score mlp: {:.2%}".format(mlp.score(Xtest, ytest)))

        count = 0
        onepreddf = pd.DataFrame()
        for x in range(len(Xtest)):
            pred =  lg.predict(Xtest[x:x + 1])
            if pred == 1:  #to search 1 cause these are the ctually appendicitis, need to convert into 0
                onepreddf = pd.concat([onepreddf, Xtest[x : x+1]], ignore_index=True, axis=0)
                count += 1
        onepreddf.to_csv(path + '' + xtest_2test + '' + '.csv')
        Xtrain.to_csv(path + '' + xtr + '' + '.csv')
        Xtest.to_csv(path + '' + xts + '' + '.csv')
        ytrain.to_csv(path + '' + ytr + '' + '.csv')
        ytest.to_csv(path + '' + yts + '' + '.csv')
        # save the model to disk
        filename = 'lg_model_mammo.sav'
        pickle.dump(lg, open(filename, 'wb'))
        print(len(Xtest), count)
        return lg, Xtest

    def make_interval_faithful(self, interval={}, dataset=''):
        if dataset == 'credit':

            for feature in interval:
                if interval[feature][1] > self.min_max_values_credit[feature]['max'] and interval[feature][0] < self.min_max_values_credit[feature]['min']:
                    interval[feature][1] = self.min_max_values_credit[feature]['max']
                    interval[feature][0] = self.min_max_values_credit[feature]['min']
                elif interval[feature][0] < self.min_max_values_credit[feature]['min']:
                    interval[feature][0] = self.min_max_values_credit[feature]['min']
                elif interval[feature][1] > self.min_max_values_credit[feature]['max']:
                    interval[feature][1] = self.min_max_values_credit[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_credit[feature]['min']:
                        interval[feature][0] = self.min_max_values_credit[feature]['min']
                    elif interval[feature][1] < self.min_max_values_credit[feature]['max']:
                        interval[feature][1] = self.min_max_values_credit[feature]['max']
                    elif interval[feature][0] < self.min_max_values_credit[feature]['min'] and interval[feature][1] < self.min_max_values_credit[feature]['max']:
                        interval[feature][0] = self.min_max_values_credit[feature]['min']
                        interval[feature][1] = self.min_max_values_credit[feature]['max']
                    else:
                        return interval
                else:
                    return interval
            return interval
        elif dataset == 'wine':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_wine[feature]['max'] and interval[feature][0] < self.min_max_values_wine[feature]['min']:
                    interval[feature][1] = self.min_max_values_wine[feature]['max']
                    interval[feature][0] = self.min_max_values_wine[feature]['min']
                elif interval[feature][0] < self.min_max_values_wine[feature]['min']:
                    interval[feature][0] = self.min_max_values_wine[feature]['min']
                elif interval[feature][1] > self.min_max_values_wine[feature]['max']:
                    interval[feature][1] = self.min_max_values_wine[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_wine[feature]['min']:
                        interval[feature][0] = self.min_max_values_wine[feature]['min']
                    elif interval[feature][1] < self.min_max_values_wine[feature]['max']:
                        interval[feature][1] = self.min_max_values_wine[feature]['max']
                    elif interval[feature][0] < self.min_max_values_wine[feature]['min'] and interval[feature][1] < self.min_max_values_wine[feature]['max']:
                        interval[feature][0] = self.min_max_values_wine[feature]['min']
                        interval[feature][1] = self.min_max_values_wine[feature]['max']
                    else:
                        return interval
                else:
                    return interval
            return interval
        elif dataset == 'bank':
            for feature in interval:
                if interval[feature][1] > self.min_max_values[feature]['max'] and interval[feature][0] < self.min_max_values[feature]['min']:
                    interval[feature][1] = self.min_max_values[feature]['max']
                    interval[feature][0] = self.min_max_values[feature]['min']
                elif interval[feature][0] < self.min_max_values[feature]['min']:
                    interval[feature][0] = self.min_max_values[feature]['min']
                elif interval[feature][1] > self.min_max_values[feature]['max']:
                    interval[feature][1] = self.min_max_values[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values[feature]['min']:
                        interval[feature][0] = self.min_max_values[feature]['min']
                    elif interval[feature][1] < self.min_max_values[feature]['max']:
                        interval[feature][1] = self.min_max_values[feature]['max']
                    elif interval[feature][0] < self.min_max_values[feature]['min'] and interval[feature][1] < self.min_max_values[feature]['max']:
                        interval[feature][0] = self.min_max_values[feature]['min']
                        interval[feature][1] = self.min_max_values[feature]['max']
                    else:
                        return interval
                else:
                    return interval
            return interval
        elif dataset == 'compass':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_compass[feature]['max'] and interval[feature][0] < self.min_max_values_compass[feature]['min']:
                    interval[feature][1] = self.min_max_values_compass[feature]['max']
                    interval[feature][0] = self.min_max_values_compass[feature]['min']
                elif interval[feature][0] < self.min_max_values_compass[feature]['min']:
                    interval[feature][0] = self.min_max_values_compass[feature]['min']
                elif interval[feature][1] > self.min_max_values_compass[feature]['max']:
                    interval[feature][1] = self.min_max_values_compass[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_compass[feature]['min']:
                        interval[feature][0] = self.min_max_values_compass[feature]['min']
                    elif interval[feature][1] < self.min_max_values_compass[feature]['max']:
                        interval[feature][1] = self.min_max_values_compass[feature]['max']
                    elif interval[feature][0] < self.min_max_values_compass[feature]['min'] and interval[feature][1] < self.min_max_values_compass[feature]['max']:
                        interval[feature][0] = self.min_max_values_compass[feature]['min']
                        interval[feature][1] = self.min_max_values_compass[feature]['max']
                    else:
                        return interval
                else:
                    return interval
            return interval
        elif dataset == 'diabetes':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_diabetes[feature]['max'] and interval[feature][0] < self.min_max_values_diabetes[feature]['min']:
                    interval[feature][1] = self.min_max_values_diabetes[feature]['max']
                    interval[feature][0] = self.min_max_values_diabetes[feature]['min']
                elif interval[feature][0] < self.min_max_values_diabetes[feature]['min']:
                    interval[feature][0] = self.min_max_values_diabetes[feature]['min']
                elif interval[feature][1] > self.min_max_values_diabetes[feature]['max']:
                    interval[feature][1] = self.min_max_values_diabetes[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_diabetes[feature]['min']:
                        interval[feature][0] = self.min_max_values_diabetes[feature]['min']
                    elif interval[feature][1] < self.min_max_values_diabetes[feature]['max']:
                        interval[feature][1] = self.min_max_values_diabetes[feature]['max']
                    elif interval[feature][0] < self.min_max_values_diabetes[feature]['min'] and interval[feature][1] < self.min_max_values_diabetes[feature]['max']:
                        interval[feature][0] = self.min_max_values_diabetes[feature]['min']
                        interval[feature][1] = self.min_max_values_diabetes[feature]['max']
                    else:
                        return interval
                else:
                    return interval
            return interval
        elif dataset == 'wisconsin':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_wisconsin[feature]['max'] and interval[feature][0] < self.min_max_values_wisconsin[feature]['min']:
                    interval[feature][1] = self.min_max_values_wisconsin[feature]['max']
                    interval[feature][0] = self.min_max_values_wisconsin[feature]['min']
                elif interval[feature][0] < self.min_max_values_wisconsin[feature]['min']:
                    interval[feature][0] = self.min_max_values_wisconsin[feature]['min']
                elif interval[feature][1] > self.min_max_values_wisconsin[feature]['max']:
                    interval[feature][1] = self.min_max_values_wisconsin[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_wisconsin[feature]['min']:
                        interval[feature][0] = self.min_max_values_wisconsin[feature]['min']
                    elif interval[feature][1] < self.min_max_values_wisconsin[feature]['max']:
                        interval[feature][1] = self.min_max_values_wisconsin[feature]['max']
                    elif interval[feature][0] < self.min_max_values_wisconsin[feature]['min'] and interval[feature][1] < self.min_max_values_wisconsin[feature]['max']:
                        interval[feature][0] = self.min_max_values_wisconsin[feature]['min']
                        interval[feature][1] = self.min_max_values_wisconsin[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_wisconsin[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'bupa':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_bupa[feature]['max'] and interval[feature][0] < self.min_max_values_bupa[feature]['min']:
                    interval[feature][1] = self.min_max_values_bupa[feature]['max']
                    interval[feature][0] = self.min_max_values_bupa[feature]['min']
                elif interval[feature][0] < self.min_max_values_bupa[feature]['min']:
                    interval[feature][0] = self.min_max_values_bupa[feature]['min']
                elif interval[feature][1] > self.min_max_values_bupa[feature]['max']:
                    interval[feature][1] = self.min_max_values_bupa[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_bupa[feature]['min']:
                        interval[feature][0] = self.min_max_values_bupa[feature]['min']
                    elif interval[feature][1] < self.min_max_values_bupa[feature]['max']:
                        interval[feature][1] = self.min_max_values_bupa[feature]['max']
                    elif interval[feature][0] < self.min_max_values_bupa[feature]['min'] and interval[feature][1] < self.min_max_values_bupa[feature]['max']:
                        interval[feature][0] = self.min_max_values_bupa[feature]['min']
                        interval[feature][1] = self.min_max_values_bupa[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_bupa[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'appendicitis':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_appendicitis[feature]['max'] and interval[feature][0] < self.min_max_values_appendicitis[feature]['min']:
                    interval[feature][1] = self.min_max_values_appendicitis[feature]['max']
                    interval[feature][0] = self.min_max_values_appendicitis[feature]['min']
                elif interval[feature][0] < self.min_max_values_appendicitis[feature]['min']:
                    interval[feature][0] = self.min_max_values_appendicitis[feature]['min']
                elif interval[feature][1] > self.min_max_values_appendicitis[feature]['max']:
                    interval[feature][1] = self.min_max_values_appendicitis[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_appendicitis[feature]['min']:
                        interval[feature][0] = self.min_max_values_appendicitis[feature]['min']
                    elif interval[feature][1] < self.min_max_values_appendicitis[feature]['max']:
                        interval[feature][1] = self.min_max_values_appendicitis[feature]['max']
                    elif interval[feature][0] < self.min_max_values_appendicitis[feature]['min'] and interval[feature][1] < self.min_max_values_appendicitis[feature]['max']:
                        interval[feature][0] = self.min_max_values_appendicitis[feature]['min']
                        interval[feature][1] = self.min_max_values_appendicitis[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_appendicitis[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'saheart':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_saheart[feature]['max'] and interval[feature][0] < self.min_max_values_saheart[feature]['min']:
                    interval[feature][1] = self.min_max_values_saheart[feature]['max']
                    interval[feature][0] = self.min_max_values_saheart[feature]['min']
                elif interval[feature][0] < self.min_max_values_saheart[feature]['min']:
                    interval[feature][0] = self.min_max_values_saheart[feature]['min']
                elif interval[feature][1] > self.min_max_values_saheart[feature]['max']:
                    interval[feature][1] = self.min_max_values_saheart[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_saheart[feature]['min']:
                        interval[feature][0] = self.min_max_values_saheart[feature]['min']
                    elif interval[feature][1] < self.min_max_values_saheart[feature]['max']:
                        interval[feature][1] = self.min_max_values_saheart[feature]['max']
                    elif interval[feature][0] < self.min_max_values_saheart[feature]['min'] and interval[feature][1] < self.min_max_values_saheart[feature]['max']:
                        interval[feature][0] = self.min_max_values_saheart[feature]['min']
                        interval[feature][1] = self.min_max_values_saheart[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_saheart[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'spotify':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_spotify[feature]['max'] and interval[feature][0] < self.min_max_values_spotify[feature]['min']:
                    interval[feature][1] = self.min_max_values_spotify[feature]['max']
                    interval[feature][0] = self.min_max_values_spotify[feature]['min']
                elif interval[feature][0] < self.min_max_values_spotify[feature]['min']:
                    interval[feature][0] = self.min_max_values_spotify[feature]['min']
                elif interval[feature][1] > self.min_max_values_spotify[feature]['max']:
                    interval[feature][1] = self.min_max_values_spotify[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_spotify[feature]['min']:
                        interval[feature][0] = self.min_max_values_spotify[feature]['min']
                    elif interval[feature][1] < self.min_max_values_spotify[feature]['max']:
                        interval[feature][1] = self.min_max_values_spotify[feature]['max']
                    elif interval[feature][0] < self.min_max_values_spotify[feature]['min'] and interval[feature][1] < self.min_max_values_spotify[feature]['max']:
                        interval[feature][0] = self.min_max_values_spotify[feature]['min']
                        interval[feature][1] = self.min_max_values_spotify[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_spotify[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'movie':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_movie[feature]['max'] and interval[feature][0] < self.min_max_values_movie[feature]['min']:
                    interval[feature][1] = self.min_max_values_movie[feature]['max']
                    interval[feature][0] = self.min_max_values_movie[feature]['min']
                elif interval[feature][0] < self.min_max_values_movie[feature]['min']:
                    interval[feature][0] = self.min_max_values_movie[feature]['min']
                elif interval[feature][1] > self.min_max_values_movie[feature]['max']:
                    interval[feature][1] = self.min_max_values_movie[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_movie[feature]['min']:
                        interval[feature][0] = self.min_max_values_movie[feature]['min']
                    elif interval[feature][1] < self.min_max_values_movie[feature]['max']:
                        interval[feature][1] = self.min_max_values_movie[feature]['max']
                    elif interval[feature][0] < self.min_max_values_movie[feature]['min'] and interval[feature][1] < self.min_max_values_movie[feature]['max']:
                        interval[feature][0] = self.min_max_values_movie[feature]['min']
                        interval[feature][1] = self.min_max_values_movie[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_movie[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'heart':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_heart[feature]['max'] and interval[feature][0] < self.min_max_values_heart[feature]['min']:
                    interval[feature][1] = self.min_max_values_heart[feature]['max']
                    interval[feature][0] = self.min_max_values_heart[feature]['min']
                elif interval[feature][0] < self.min_max_values_heart[feature]['min']:
                    interval[feature][0] = self.min_max_values_heart[feature]['min']
                elif interval[feature][1] > self.min_max_values_heart[feature]['max']:
                    interval[feature][1] = self.min_max_values_heart[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_heart[feature]['min']:
                        interval[feature][0] = self.min_max_values_heart[feature]['min']
                    elif interval[feature][1] < self.min_max_values_heart[feature]['max']:
                        interval[feature][1] = self.min_max_values_heart[feature]['max']
                    elif interval[feature][0] < self.min_max_values_heart[feature]['min'] and interval[feature][1] < self.min_max_values_heart[feature]['max']:
                        interval[feature][0] = self.min_max_values_heart[feature]['min']
                        interval[feature][1] = self.min_max_values_heart[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_heart[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'wdbc':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_wdbc[feature]['max'] and interval[feature][0] < self.min_max_values_wdbc[feature]['min']:
                    interval[feature][1] = self.min_max_values_wdbc[feature]['max']
                    interval[feature][0] = self.min_max_values_wdbc[feature]['min']
                elif interval[feature][0] < self.min_max_values_wdbc[feature]['min']:
                    interval[feature][0] = self.min_max_values_wdbc[feature]['min']
                elif interval[feature][1] > self.min_max_values_wdbc[feature]['max']:
                    interval[feature][1] = self.min_max_values_wdbc[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_wdbc[feature]['min']:
                        interval[feature][0] = self.min_max_values_wdbc[feature]['min']
                    elif interval[feature][1] < self.min_max_values_wdbc[feature]['max']:
                        interval[feature][1] = self.min_max_values_wdbc[feature]['max']
                    elif interval[feature][0] < self.min_max_values_wdbc[feature]['min'] and interval[feature][1] < self.min_max_values_wdbc[feature]['max']:
                        interval[feature][0] = self.min_max_values_wdbc[feature]['min']
                        interval[feature][1] = self.min_max_values_wdbc[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_wdbc[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'magic':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_magic[feature]['max'] and interval[feature][0] < self.min_max_values_magic[feature]['min']:
                    interval[feature][1] = self.min_max_values_magic[feature]['max']
                    interval[feature][0] = self.min_max_values_magic[feature]['min']
                elif interval[feature][0] < self.min_max_values_magic[feature]['min']:
                    interval[feature][0] = self.min_max_values_magic[feature]['min']
                elif interval[feature][1] > self.min_max_values_magic[feature]['max']:
                    interval[feature][1] = self.min_max_values_magic[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_magic[feature]['min']:
                        interval[feature][0] = self.min_max_values_magic[feature]['min']
                    elif interval[feature][1] < self.min_max_values_magic[feature]['max']:
                        interval[feature][1] = self.min_max_values_magic[feature]['max']
                    elif interval[feature][0] < self.min_max_values_magic[feature]['min'] and interval[feature][1] < self.min_max_values_magic[feature]['max']:
                        interval[feature][0] = self.min_max_values_magic[feature]['min']
                        interval[feature][1] = self.min_max_values_magic[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_magic[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'titanic':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_titanic[feature]['max'] and interval[feature][0] < self.min_max_values_titanic[feature]['min']:
                    interval[feature][1] = self.min_max_values_titanic[feature]['max']
                    interval[feature][0] = self.min_max_values_titanic[feature]['min']
                elif interval[feature][0] < self.min_max_values_titanic[feature]['min']:
                    interval[feature][0] = self.min_max_values_titanic[feature]['min']
                elif interval[feature][1] > self.min_max_values_titanic[feature]['max']:
                    interval[feature][1] = self.min_max_values_titanic[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_titanic[feature]['min']:
                        interval[feature][0] = self.min_max_values_titanic[feature]['min']
                    elif interval[feature][1] < self.min_max_values_titanic[feature]['max']:
                        interval[feature][1] = self.min_max_values_titanic[feature]['max']
                    elif interval[feature][0] < self.min_max_values_titanic[feature]['min'] and interval[feature][1] < self.min_max_values_titanic[feature]['max']:
                        interval[feature][0] = self.min_max_values_titanic[feature]['min']
                        interval[feature][1] = self.min_max_values_titanic[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_titanic[feature]['min']
                else:
                    return interval
            return interval
        elif dataset == 'mammo':
            for feature in interval:
                if interval[feature][1] > self.min_max_values_mammo[feature]['max'] and interval[feature][0] < self.min_max_values_mammo[feature]['min']:
                    interval[feature][1] = self.min_max_values_mammo[feature]['max']
                    interval[feature][0] = self.min_max_values_mammo[feature]['min']
                elif interval[feature][0] < self.min_max_values_mammo[feature]['min']:
                    interval[feature][0] = self.min_max_values_mammo[feature]['min']
                elif interval[feature][1] > self.min_max_values_mammo[feature]['max']:
                    interval[feature][1] = self.min_max_values_mammo[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_mammo[feature]['min']:
                        interval[feature][0] = self.min_max_values_mammo[feature]['min']
                    elif interval[feature][1] < self.min_max_values_mammo[feature]['max']:
                        interval[feature][1] = self.min_max_values_mammo[feature]['max']
                    elif interval[feature][0] < self.min_max_values_mammo[feature]['min'] and interval[feature][1] < self.min_max_values_mammo[feature]['max']:
                        interval[feature][0] = self.min_max_values_mammo[feature]['min']
                        interval[feature][1] = self.min_max_values_mammo[feature]['max']
                    else:
                        return interval
                elif interval[feature][0] < 0:
                    interval[feature][0] = self.min_max_values_mammo[feature]['min']
                else:
                    return interval
            return interval
        else: #adult
            for feature in interval:
                if interval[feature][1] > self.min_max_values_adult[feature]['max'] and interval[feature][0] < self.min_max_values_adult[feature]['min']:
                    interval[feature][1] = self.min_max_values_adult[feature]['max']
                    interval[feature][0] = self.min_max_values_adult[feature]['min']
                elif interval[feature][0] < self.min_max_values_adult[feature]['min']:
                    interval[feature][0] = self.min_max_values_adult[feature]['min']
                elif interval[feature][1] > self.min_max_values_adult[feature]['max']:
                    interval[feature][1] = self.min_max_values_adult[feature]['max']
                elif interval[feature][0] < 0 and interval[feature][1] < 0:
                    if interval[feature][0] < self.min_max_values_adult[feature]['min']:
                        interval[feature][0] = self.min_max_values_adult[feature]['min']
                    elif interval[feature][1] < self.min_max_values_adult[feature]['max']:
                        interval[feature][1] = self.min_max_values_adult[feature]['max']
                    elif interval[feature][0] < self.min_max_values_adult[feature]['min'] and interval[feature][1] < self.min_max_values_adult[feature]['max']:
                        interval[feature][0] = self.min_max_values_adult[feature]['min']
                        interval[feature][1] = self.min_max_values_adult[feature]['max']
                    else:
                        return interval
                else:
                    return interval
            return interval


    def train_Outliers_isolation_model(self, df):
        #import matplotlib.pyplot as plt
        from sklearn.ensemble import IsolationForest
        ##and look at the data
        df1 = df.copy()
        #plt.figure(figsize=(20, 10))
        #plt.scatter(df1['Income'], df1['Mortgage'])
        #plt.show()
        ##apply an Isolation forest
        outlier_model = IsolationForest(n_estimators=100, max_samples=1000, contamination=.05, max_features=df1.shape[1])
        outlier_model.fit(df1)
        outliers_predicted = outlier_model.predict(df1)

        # check the results
        #df1['outlier'] = outliers_predicted
        #plt.figure(figsize=(20, 10))
        #plt.scatter(df1['Income'], df1['Mortgage'], c=df1['outlier'])
        #plt.show()
        return outlier_model

    def get_Outlier_isolation_prediction(self, model, cf_instance):
        predicted = model.predict(cf_instance)
        print(predicted)

    def MahalanobisDist_outlier_model(self, df, verbose=False):
        import numpy as np
        #data = df.as_matrix(columns=None)
        data = df.values
        covariance_matrix = np.cov(data, rowvar=False)
        if self.is_pos_def(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if self.is_pos_def(inv_covariance_matrix):
                vars_mean = []
                for i in range(data.shape[0]):
                    vars_mean.append(list(data.mean(axis=0)))
                diff = data - vars_mean
                md = []
                for i in range(len(diff)):
                    md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))

                if verbose:
                    print("Covariance Matrix:\n {}\n".format(covariance_matrix))
                    print("Inverse of Covariance Matrix:\n {}\n".format(inv_covariance_matrix))
                    print("Variables Mean Vector:\n {}\n".format(vars_mean))
                    print("Variables - Variables Mean Vector:\n {}\n".format(diff))
                    print("Mahalanobis Distance:\n {}\n".format(md))
                return md
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
            print("Error: Covariance Matrix is not positive definite!")

    def is_pos_def(self, A):
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def MD_removeOutliers(self, df):
        MD = self.MahalanobisDist_outlier_model(df, verbose=False)
        threshold = np.mean(MD) * 2.2  # adjust 1.5 accordingly
        outliers = []
        for i in range(len(MD)):
            if MD[i] > threshold:
                outliers.append(i)  # index of the outlier
        return np.array(outliers)


    def features_to_explain(self, test_df, cfs_df):
        #cfs_df.sort_values(by=cfs_df.columns[0], ascending=True)
        feature_list = []
        new_values = []
        actual_values = []
        for f in cfs_df.columns:
            if cfs_df[f].values != test_df[f].values:
                feature_list.append(f)
                new_values.append(cfs_df[f].values[0])
                actual_values.append(test_df[f].values[0])
        return feature_list, new_values, actual_values

    def generate_reason_explanation(self, outcome_variable, desired_class, actual_class, features):
        lexicon = Lexicon.getDefaultLexicon()
        nlgFactory = NLGFactory(lexicon)
        realiser = Realiser(lexicon)
        print("OUTCOME REASONS:")
        # phrase part 1
        subj = outcome_variable  # "loan" could be dynamic
        # subj.setDeterminer("the")
        verb = "be"
        obj = actual_class  # yes or no, need to change dynamically
        phrase_part1 = nlgFactory.createClause()  # "sentence part 1"
        phrase_part1.setSubject(subj)
        phrase_part1.setVerb(verb)
        phrase_part1.setObject(obj)
        # phrase part2
        # subj.setDeterminer("the")
        subject = []  # should be dynamic as per the list of features to change
        if len(features) == 1:
            subj1 = features[0]
        elif len(features) == 2:
            subject1 = nlgFactory.createNounPhrase(features[0])
            subject2 = nlgFactory.createNounPhrase(features[1])
            subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
        elif len(features) == 3:
            subject1 = nlgFactory.createNounPhrase(features[0])
            subject2 = nlgFactory.createNounPhrase(features[1])
            subject3 = nlgFactory.createNounPhrase(features[2])
            subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
            subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject3)
        elif len(features) == 4:
            subject1 = nlgFactory.createNounPhrase(features[0])
            subject2 = nlgFactory.createNounPhrase(features[1])
            subject3 = nlgFactory.createNounPhrase(features[2])
            subject4 = nlgFactory.createNounPhrase(features[3])
            subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
            subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject3)
            subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject4)
        verb1 = "be"
        obj1 = "enough"  # yes or no, need to change dynamically
        phrase_part2 = nlgFactory.createClause()  # "sentence part 1"
        phrase_part2.setSubject(subj1)
        phrase_part2.setVerb(verb1)
        phrase_part2.setObject(obj1)

        phrase_part1.setFeature(Feature.TENSE, Tense.PRESENT)
        phrase_part2.setFeature(Feature.COMPLEMENTISER, "because values of")
        phrase_part2.setFeature(Feature.NEGATED, True)
        phrase_part1.addComplement(phrase_part2)

        output = realiser.realiseSentence(phrase_part1)
        print(output)

    def generate_suggestion_explanation(self, outcome_variable, desired_class, actual_class, features, new_values, actual_values):
        lexicon = Lexicon.getDefaultLexicon()
        nlgFactory = NLGFactory(lexicon)
        realiser = Realiser(lexicon)
        print("Suggestion-Explanation:")
        # phrase part 1
        subj = outcome_variable  # "loan" could be dynamic
        verb = "would be"
        obj = desired_class  # yes or no, need to change dynamically
        phrase_part1 = nlgFactory.createClause()  # "sentence part 1"
        phrase_part1.setSubject(subj)
        phrase_part1.setVerb(verb)
        phrase_part1.setObject(obj)
        # phrase part2
        subject = []
        if len(features) == 1:
            subj1 = features[0]
        elif len(features) == 2:
            subject1 = nlgFactory.createNounPhrase(features[0])
            subject2 = nlgFactory.createNounPhrase(features[1])
            subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
        elif len(features) == 3:
            subject1 = nlgFactory.createNounPhrase(features[0])
            subject2 = nlgFactory.createNounPhrase(features[1])
            subject3 = nlgFactory.createNounPhrase(features[2])
            subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
            subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject3)
        elif len(features) == 4:
            subject1 = nlgFactory.createNounPhrase(features[0])
            subject2 = nlgFactory.createNounPhrase(features[1])
            subject3 = nlgFactory.createNounPhrase(features[2])
            subject4 = nlgFactory.createNounPhrase(features[3])
            subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
            subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject3)
            subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject4)
        verb1 = "change features"
        obj1 = "you"  # yes or no, need to change dynamically
        phrase_part2 = nlgFactory.createClause()  # "sentence part 1"
        phrase_part2.setSubject(obj1)
        phrase_part2.setVerb(verb1)
        phrase_part2.setObject(subj1)

        phrase_part1.setFeature(Feature.TENSE, Tense.PRESENT)
        phrase_part2.setFeature(Feature.COMPLEMENTISER, "if")
        phrase_part1.addComplement(phrase_part2)

        # phrase part 3
        if len(features) == 1:
            subj2 = str(new_values[0])
        elif len(features) == 2:
            object1 = nlgFactory.createNounPhrase(str(new_values[0]))
            object2 = nlgFactory.createNounPhrase(str(new_values[1]))
            subj2 = nlgFactory.createCoordinatedPhrase(object1, object2)
        elif len(features) == 3:
            object1 = nlgFactory.createNounPhrase(str(new_values[0]))
            object2 = nlgFactory.createNounPhrase(str(new_values[1]))
            object3 = nlgFactory.createNounPhrase(str(new_values[2]))
            subj2 = nlgFactory.createCoordinatedPhrase(object1, object2)
            subj2 = nlgFactory.createCoordinatedPhrase(subj2, object3)
        elif len(features) == 4:
            object1 = nlgFactory.createNounPhrase(str(new_values[0]))
            object2 = nlgFactory.createNounPhrase(str(new_values[1]))
            object3 = nlgFactory.createNounPhrase(str(new_values[2]))
            object4 = nlgFactory.createNounPhrase(str(new_values[3]))
            subj2 = nlgFactory.createCoordinatedPhrase(object1, object2)
            subj2 = nlgFactory.createCoordinatedPhrase(subj2, object3)
            subj2 = nlgFactory.createCoordinatedPhrase(subj2, object4)
        verb2 = "change features"
        obj2 = "you"  # yes or no, need to change dynamically
        phrase_part3 = nlgFactory.createClause()  # "sentence part 1"
        phrase_part3.setSubject(subj2)

        phrase_part2.setFeature(Feature.TENSE, Tense.PRESENT)
        phrase_part3.setFeature(Feature.COMPLEMENTISER, "to")
        phrase_part1.addComplement(phrase_part3)

        output = realiser.realiseSentence(phrase_part1)
        print(output)

    # def verify_causal_realistic_relations(self):
    #     """"
    #      list of hand-crafted rules for all the features in the dataset
    #      if edu > actual_edu then age > actual_age as per the requirements etc.
    #     """
    def potential_cfs(self, val=10):
        self.testvalue =  val
        print("test val ufce:", self.testvalue)

    def euclidean_Dist(df1, df2, cols=['Age','Experience']):
        return np.linalg.norm(df1[cols].values - df2[cols].values, axis=0)
    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def plot_heatmap(self, df, cols, vals):
        import seaborn as sns
        sns.heatmap(df, cols, vals)
