import utilites
import torch
import numpy as np
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.nn import Module
import numpy as np
import copy
from typing import Callable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import defaultdict
#from __future__ import division
from sklearn import preprocessing
from collections import defaultdict
import urllib
import sys, os
import numpy as np
import math
import random
import itertools
import copy
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


def check_data_file(fname):
    files = os.listdir(".") # get the current directory listing
    print("Looking for file in the current directory...")

    if fname not in files:
        print(" file not found! Downloading from GitHub...")
        #/propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv
        addr = "https://raw.githubusercontent.com//propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv"
        response = urllib.request.urlopen(addr)
        data = response.read()
        fileOut = open(fname, "wb")
        fileOut.write(data)
        fileOut.close()
        print("download and saved locally..")
    else:
        print("File found in current directory..")






def sensr_adult_preprocess(root_path : str):
    ADULT_COLUMN_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-classification']

    CONTINUOUS_FEATURES = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    if root_path[-1] != '/' and root_path[-1] != '\\':
        root_path += '/'
    train = pd.read_csv(root_path + 'adult.data', names=ADULT_COLUMN_NAMES, sep=',\s', engine='python')
    test = pd.read_csv(root_path + 'adult.test', names=ADULT_COLUMN_NAMES, sep=',\s', engine='python', header=1)

    df = pd.concat([train, test], ignore_index=True)

    df = df.replace('?', np.NaN).dropna()

    df['income-classification'] = df['income-classification'].replace({'<=50K.': 0, '>50K.' : 1, '>50K' : 1, '<=50K' : 0})
    df = pd.get_dummies(df, columns=['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])


    delete_these = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other', 'sex_Female']

    delete_these += ['native-country_Cambodia', 'native-country_Canada', 'native-country_China',
                     'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 
                     'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England',
                     'native-country_France', 'native-country_Germany', 'native-country_Greece', 
                     'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands',
                     'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 
                     'native-country_India', 'native-country_Iran', 'native-country_Ireland', 
                     'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 
                     'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 
                     'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 
                     'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal',
                     'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South',
                     'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 
                     'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia']

    delete_these += ['fnlwgt', 'education']

    df.drop(delete_these, axis=1, inplace=True)

    for col in CONTINUOUS_FEATURES:
        #normalize continuous features
        df[col] = (df[col] -  df[col].mean()) / (df[col].std())

    classification = df['income-classification']
    df = df.drop('income-classification', axis=1)
    df.insert(loc=len(df.columns), column='income-classification', value=classification)
    for col in df.columns:
        df[col] = df[col].astype(np.float32)

    return df



def get_adult():



    class SensrAdultDataset(Dataset):
        def __init__(self, adult_root_path,protected_att = "sex_Male"):
            super().__init__()
            self.pandas_rep = sensr_adult_preprocess(adult_root_path)
            self.index = sensr_adult_preprocess(adult_root_path).columns.get_loc(protected_att)
            self.x = torch.tensor((self.pandas_rep.loc[:, self.pandas_rep.columns != 'income-classification']).to_numpy(np.float32), dtype=torch.float32)
            self.y = torch.tensor((self.pandas_rep.loc[:, 'income-classification']).to_numpy(np.float32), dtype=torch.float32)

        def __len__(self):
            return len(self.pandas_rep)

        def get_columns(self):
            return self.pandas_rep.columns

        def __getitem__(self, index):
            inputs = self.x[index, :]
            if len(self.y.shape) == 1:
                targets = self.y[index]
            elif len(self.y.shape == 2):
                targets = self.y[index, :]
            if len(targets.shape) < 2:
                targets = torch.unsqueeze(targets, len(targets.shape))
            return inputs, targets
    dataset = SensrAdultDataset('./Raw_Data/Adult/')
    
    return dataset
    
    
    
    
    
def get_compass():
    COMPAS_INPUT_FILE = "compas-scores-two-years.csv"
    check_data_file(COMPAS_INPUT_FILE)
    FEATURES = ['Two_yr_Recidivism', 'Number_of_Priors', 'score_factor',
       'Age_Above_FourtyFive', 'Age_Below_TwentyFive', 'African_American',
       'Asian', 'Hispanic', 'Native_American', 'Other', 'Female',
       'Misdemeanor']
    FEATURES_CLASS = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"] 
    CONTI_FEATURE = ["priors_count"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid" # the classifier variable
    SENSITIVE_FEATURE = "race"
    df = pd.read_csv(COMPAS_INPUT_FILE)
    
    df = df.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals
    df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30) & (df.is_recid != -1) & (df.c_charge_degree != 'O') & (df.score_text != 'N/A') ]
    df.reset_index(inplace=True, drop=True) # renumber the rows from 0 
    
    #Convert to NP Array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])
    
    
    #Normalize the feature
    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    #y[y==0] = -1


        # print ("\nNumber of people recidivating within two years")
        # print (pd.Series(y).value_counts()) #-1 mean were not rearrested
        # print ("\n")

    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
    #print(X.shape)
    x_control = defaultdict(list)


    feature_names = []

    for attr in FEATURES_CLASS:
        vals = data[attr]
        if attr in CONTI_FEATURE:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals) # 0 mean and 1 variance  
            vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col
        else: # for binary categorical variables, the label binarizer uses just one var instead of two
           lb = preprocessing.LabelBinarizer()
           lb.fit(vals)
           vals = lb.transform(vals)
        if attr in SENSITIVE_FEATURE:
            x_control[attr] = vals


		# add to learnable features
        X = np.hstack((X, vals))
        if attr in CONTI_FEATURE: # continuous feature, just append the name
           feature_names.append(attr)
        else: # categorical features
           if vals.shape[1] == 1: # binary features that passed through lib binarizer
              feature_names.append(attr)
           else:
             for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                feature_names.append(attr + "_" + str(k))


    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
		#assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()
        
    X = torch.tensor(X,dtype = torch.float32)
    y = torch.tensor(y,dtype = torch.float32)
    y = y.unsqueeze(1)
    
    ##print(X.shape)
    ##rint(y.shape)
    dataset = TensorDataset(X, y)
    
    return dataset


    