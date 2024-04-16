import datasets
import torch
from utilites import get_attribute_tensor
import sys, os
import numpy as np
import math
import random
import itertools
import copy
from models import BinaryLogisticRegression
from models_fairbatch import LogisticRegression, weights_init_normal, test_model
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch
from sklearn.model_selection import train_test_split

from FairBatchSampler import FairBatch, CustomDataset

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt



desired_data = 'COMPAS'

if(desired_data == 'ADULT'):
    dataset = datasets.get_adult() ####
    protected_index = 40 ## Corresponds to sex

elif(desired_data == 'COMPAS'):
    dataset = datasets.get_compass()  ####
    protected_index = 3  ## Corresponds to African American race
else:
    pass




# Split the dataset into feature_tensors and target_tensors
feature_tensors = [batch[0] for batch in dataset]
target_tensors = [batch[1] for batch in dataset]

# Create tensors from the lists
feature_tensor = torch.stack(feature_tensors)
non_sens_feature ,sens_feature = get_attribute_tensor(feature_tensor , protected_index )
target_tensor = torch.stack(target_tensors)
## Replace labels



test_size = 0.2

# Split the data into training and testing sets
xz_train, xz_test,z_train, z_test, y_train, y_test = train_test_split(
    non_sens_feature, sens_feature ,target_tensor, test_size=test_size, random_state=42)



z_train = z_train.squeeze(1)
y_train = y_train.squeeze(1)


z_test = z_test.squeeze(1)
y_test = y_test.squeeze(1)


xz_train = torch.FloatTensor(xz_train)
y_train = torch.FloatTensor(y_train).float()
z_train = torch.FloatTensor(z_train)

xz_test = torch.FloatTensor(xz_test)
y_test = torch.FloatTensor(y_test)
z_test = torch.FloatTensor(z_test)


y_train = torch.where(y_train == 0.0, -1.0, y_train.double()).float()
y_test = torch.where(y_test == 0.0, -1.0, y_test.double()).float()


def run_epoch(model, train_features, labels, optimizer, criterion):
    """Trains the model with the given train data.

    Args:
        model: A torch model to train.
        train_features: A torch tensor indicating the train features.
        labels: A torch tensor indicating the true labels.
        optimizer: A torch optimizer.
        criterion: A torch criterion.

    Returns:
        loss value.
    """
    
    optimizer.zero_grad()

    label_predicted = model.forward(train_features)
    #label_predicted = label_predicted.squeeze()
   
    # Make sure labels have the same shape as label_predicted
    labels = labels.squeeze()
    loss  = criterion((F.tanh(label_predicted.squeeze())+1)/2, labels)  # Remove the .squeeze() for labels
    loss.backward()

    optimizer.step()
    
    return loss.item()





full_tests = []

# Set the train data
train_data = CustomDataset(xz_train, y_train, z_train)
feature_shape = xz_train.shape[1]
seeds = [0]

for seed in seeds:

    print("< Seed: {} >".format(seed))

    # ---------------------
    #  Initialize model, optimizer, and criterion
    # ---------------------

    model = LogisticRegression(feature_shape,1)

    torch.manual_seed(seed)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()

    losses = []

    # ---------------------
    #  Define FairBatch and DataLoader
    # ---------------------

    sampler = FairBatch (model, train_data.x, train_data.y, train_data.z, batch_size = 100, alpha = 0.005, target_fairness = 'original', replacement = False)
    train_loader = torch.utils.data.DataLoader (train_data, sampler=sampler, num_workers=0)

    # ---------------------
    #  Model training
    # ---------------------
    for epoch in range(400):
        print(epoch)
        tmp_loss = []

        for batch_idx, (data, target, z) in enumerate (train_loader):
            loss = run_epoch (model, data, target, optimizer, criterion)
            tmp_loss.append(loss)

        losses.append(sum(tmp_loss)/len(tmp_loss))

    tmp_test = test_model(model, xz_test, y_test, z_test)
    full_tests.append(tmp_test)

    vanilla_acc = tmp_test['Acc']
    vanilla_eo = tmp_test['EO_Y1_diff']
    vanilla_eqodds = tmp_test['EqOdds_diff']
    vanilla_dp = tmp_test['DP_diff']

    print("  Test accuracy: {}".format(vanilla_acc))
    print("  EO disparity: {}".format(vanilla_eo))
    print("  EqOdds disparity: {}".format(vanilla_eqodds))
    print("  DP disparity: {}".format(vanilla_dp))
    print("----------------------------------------------------------------------")
    
    
     
full_tests = []

# Set the train data
train_data = CustomDataset(xz_train, y_train, z_train)


seeds = [0]
for seed in seeds:

    print("< Seed: {} >".format(seed))

    # ---------------------
    #  Initialize model, optimizer, and criterion
    # ---------------------

    model =  LogisticRegression(feature_shape,1)

    torch.manual_seed(seed)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()

    losses = []

    # ---------------------
    #  Define FairBatch and DataLoader
    # ---------------------

    sampler = FairBatch (model, train_data.x, train_data.y, train_data.z, batch_size = 100, alpha = 0.005, target_fairness = 'eqopp', replacement = False, seed = seed)
    train_loader = torch.utils.data.DataLoader (train_data, sampler=sampler, num_workers=0)

    # ---------------------
    #  Model training
    # ---------------------
    for epoch in range(300):

        tmp_loss = []

        for batch_idx, (data, target, z) in enumerate (train_loader):
            loss = run_epoch (model, data, target, optimizer, criterion)
            tmp_loss.append(loss)

        losses.append(sum(tmp_loss)/len(tmp_loss))

    tmp_test = test_model(model, xz_test, y_test, z_test)
    full_tests.append(tmp_test)

    fairbatch_1_acc = tmp_test['Acc']
    fairbatch_1_eo = tmp_test['EO_Y1_diff']

    print("  Test accuracy: {}".format(fairbatch_1_acc))
    print("  EO disparity: {}".format(fairbatch_1_eo))
    print("----------------------------------------------------------------------")



"""### Equalized odds"""

full_tests = []

# Set the train data
train_data = CustomDataset(xz_train, y_train, z_train)

seeds = [0]
for seed in seeds:

    print("< Seed: {} >".format(seed))

    # ---------------------
    #  Initialize model, optimizer, and criterion
    # ---------------------

    model =  LogisticRegression(feature_shape,1)

    torch.manual_seed(seed)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()

    losses = []

    # ---------------------
    #  Define FairBatch and DataLoader
    # ---------------------

    sampler = FairBatch (model, train_data.x, train_data.y, train_data.z, batch_size = 100, alpha = 0.005, target_fairness = 'eqodds', replacement = False, seed = seed)
    train_loader = torch.utils.data.DataLoader (train_data, sampler=sampler, num_workers=0)

    # ---------------------
    #  Model training
    # ---------------------
    for epoch in range(400):

        tmp_loss = []

        for batch_idx, (data, target, z) in enumerate (train_loader):
            loss = run_epoch (model, data, target, optimizer, criterion)
            tmp_loss.append(loss)

        losses.append(sum(tmp_loss)/len(tmp_loss))

    tmp_test = test_model(model, xz_test, y_test, z_test)
    full_tests.append(tmp_test)

    fairbatch_2_acc = tmp_test['Acc']
    fairbatch_2_eqodds = tmp_test['EqOdds_diff']

    print("  Test accuracy: {}".format(fairbatch_2_acc))
    print("  EqOdds disparity: {}".format(fairbatch_2_eqodds))
    print("----------------------------------------------------------------------")

"""### Demographic parity"""

full_tests = []

# Set the train data
train_data = CustomDataset(xz_train, y_train, z_train)

seeds = [0]
for seed in seeds:

    print("< Seed: {} >".format(seed))

    # ---------------------
    #  Initialize model, optimizer, and criterion
    # ---------------------

    model =  LogisticRegression(feature_shape,1)

    torch.manual_seed(seed)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()

    losses = []

    # ---------------------
    #  Define FairBatch and DataLoader
    # ---------------------

    sampler = FairBatch (model, train_data.x, train_data.y, train_data.z, batch_size = 100, alpha = 0.005, target_fairness = 'dp', replacement = False, seed = seed)
    train_loader = torch.utils.data.DataLoader (train_data, sampler=sampler, num_workers=0)

    # ---------------------
    #  Model training
    # ---------------------
    for epoch in range(450):

        tmp_loss = []

        for batch_idx, (data, target, z) in enumerate (train_loader):
            loss = run_epoch (model, data, target, optimizer, criterion)
            tmp_loss.append(loss)

        losses.append(sum(tmp_loss)/len(tmp_loss))

    tmp_test = test_model(model, xz_test, y_test, z_test)
    full_tests.append(tmp_test)

    fairbatch_3_acc = tmp_test['Acc']
    fairbatch_3_dp = tmp_test['DP_diff']

    print("  Test accuracy: {}".format(fairbatch_3_acc))
    print("  DP disparity: {}".format(fairbatch_3_dp))
    print("----------------------------------------------------------------------")






"""## **Part IV: Evaluation**

We now observe how much FairBatch mitigates the unfairness of the model.
"""

xticks = ('  Equal Opportunity', 'Equalized Odds', 'Demographic Parity')
index = np.arange(len(xticks))

vanilla_plot = (vanilla_eo, vanilla_eqodds, vanilla_dp)
fairbatch_plot = (fairbatch_1_eo, fairbatch_2_eqodds, fairbatch_3_dp)

ind = np.arange(len(xticks))  # the x locations for the groups
width = 0.27  # the width of the bars

fig, ax = plt.subplots(1,1,figsize=(13,7))
rects1 = ax.bar(ind - width/2, vanilla_plot, width, label='Vanilla (non-fair)', color = 'salmon')
rects2 = ax.bar(ind  + width/2, fairbatch_plot, width, label='FairBatch', color = 'royalblue')

plt.ylabel('Unfairness', fontsize=40)
plt.xticks(ind, xticks, fontsize = 35, rotation = 0)
ax.legend()

plt.tick_params(labelsize=27) #27

plt.legend(loc='upper center', fontsize=30, edgecolor='black',framealpha=0, ncol=5, bbox_to_anchor=(0.5, 1.15)).get_frame().set_linewidth(1.5) #25
plt.gcf().subplots_adjust(bottom=0.17)

# plt.xticks(rotation=10)
plt.tight_layout()
fig1 = plt.gcf()
plt.show()   


