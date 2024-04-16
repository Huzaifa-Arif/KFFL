import matplotlib.pyplot as plt
import pickle

import os


#filename = "test_results.pickle"

### Convergent Acc and SPD ####
def get_conv_acc_spd(filename):
    print(filename)
    with open(filename, "rb") as file:
        results = pickle.load(file)
        results_std = pickle.load(file)
        dataset = pickle.load(file)
    acc = list(results['Acc'])
    spd = list(results['SPD'])
    eod = list(results['EOD'])
    
    acc_std = list(results_std['Acc'])
    spd_std = list(results_std['SPD'])
    eod_std = list(results_std['EOD'])
    #return min(acc),min(spd),min(eod),acc_std[-1],spd_std[-1],eod_std[-1]
    return acc[-1],spd[-1],eod[-1],acc_std[-1],spd_std[-1],eod_std[-1]


def get_all_comcost(filename):
    print(filename)
    with open(filename, "rb") as file:
        results = pickle.load(file)
        results_std = pickle.load(file)
        dataset = pickle.load(file)
    comm_cost = list(results['Comm_Cost'])
  
    return comm_cost

def get_all_acc_spd(filename):
    print(filename)
    with open(filename, "rb") as file:
        results = pickle.load(file)
        results_std = pickle.load(file)
        dataset = pickle.load(file)
    acc = list(results['Acc'])
    spd = list(results['SPD'])
    return acc,spd
    

def get_limits(dataset):
    if(dataset == 'COMPAS'):
        acclim1 = 0.0
        acclim2 = 0.7
        spdlim1 = -0.2
        spdlim2 = 1.0
    
    elif(dataset == 'ADULT'):
        acclim1 = 0.0
        acclim2 = 0.85
        spdlim1 = -0.2
        spdlim2 = 1.0
    
    else:
        print('Invalid Dataset')
    return acclim1,acclim2, spdlim1,spdlim2
    
    
    
def plot_individual(filename):
    print(filename)
    with open(filename, "rb") as file:
        results = pickle.load(file)
        results_std = pickle.load(file)
        dataset = pickle.load(file)
    
    #results = {'Acc' : acc, 'SPD': spd, 'EOD' : eod, 'Comm_Cost' : comm_cost}
    # Sample data
    accuracy = results['Acc']#[0.8, 0.85, 0.9, 0.92, 0.95]
    statistical_parity  = results['SPD']#[100, 80, 60, 50, 40]
    equalized_odds = results['EOD']#[0.7, 0.75, 0.8, 0.82, 0.85]
    communication = results['Comm_Cost']#[50, 40, 30, 20, 10]
    
    accuracy_std = results_std['Acc']#[0.8, 0.85, 0.9, 0.92, 0.95]
    statistical_parity_std = results_std['SPD']#[100, 80, 60, 50, 40]
    equalized_odds_std = results_std['EOD']#[0.7, 0.75, 0.8, 0.82, 0.85]
    communication_std = results_std['Comm_Cost']#[50, 40, 30, 20, 10]
    
    
    #import pdb; pdb.set_trace()
    print(f'Verify Dataset right now {dataset}')
    print(equalized_odds_std)

    acclim1,acclim2, spdlim1,spdlim2 = get_limits(dataset)
    
    # Create the figure and subplots

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    # Accuracy vs. Cost
    #ax1.scatter(communication , accuracy)
    ax1.errorbar(communication , accuracy, yerr=accuracy_std, fmt='o', capsize=5)
    ax1.set_xlabel('Cost (MB)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(filename+'Accuracy vs. Cost')
    ax1.set_ylim(acclim1, acclim2)
    ax1.grid(True)
    # Accuracy vs. Statistical Parity
    #ax2.scatter(statistical_parity, accuracy)
    ax2.errorbar(statistical_parity , accuracy, yerr=accuracy_std, fmt='o', capsize=5)
    ax2.set_xlabel('Statistical Parity')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(filename+'Accuracy vs. Statistical Parity')
    ax2.set_ylim(acclim1, acclim2)
    ax2.set_xlim(spdlim1, spdlim2)
    ax2.grid(True)
    # Statistical Parity vs. Communication
    #ax3.scatter(communication, statistical_parity)
    ax3.errorbar(communication , accuracy, yerr=statistical_parity_std, fmt='o', capsize=5)
    ax3.set_xlabel('Communication (MB)')
    ax3.set_ylabel('Statistical Parity')
    ax3.set_title(filename + 'Statistical Parity vs. Communication')
    ax3.set_ylim(spdlim1,spdlim2)
    ax3.grid(True)
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.savefig('sample_figure.pdf')
    
    
    # Display the plot
    plt.show()
    

def get_dataset(filename):
    print(filename)
    with open(filename, "rb") as file:
        results = pickle.load(file)
        results_std = pickle.load(file)
        dataset = pickle.load(file)
    return dataset
    
    

def plot_fairness_weight_Pareto_SPD(spd, acc, acc_std, spd_std, fairness_weights, dataset,distribution,mdl,method):
    # Define markers and colors for each fairness weight (customize as needed)
    markers = ['o', 's', 'D', '^', 'v', 'p', '*']  # Added 'p' for pentagon and '*' for star
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Added 'y' for yellow and 'k' for black

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    for i, fairness_weight in enumerate(fairness_weights):
        fairness_weight = str(fairness_weight)
        plt.errorbar(eod[fairness_weight], acc[fairness_weight], xerr=eod_std[fairness_weight], yerr=acc_std[fairness_weight], 
             marker=markers[i], color=colors[i], label=f'Fairness Weight {float(fairness_weight):.4f}', linestyle='none', markersize=8)

    # Label the axes
    plt.xlabel('Statistical Parity (SPD)')
    plt.ylabel('Accuracy (ACC)')

    # Add a legend
    plt.legend()

    # Set title and show the plot
    plt.title('Pareto_SPD_' + '_'+ dataset+'_' +distribution+'_'+ mdl+'_'+ method+ '_DATASET')
    plt.grid(True)
    
    # Saving Fig
    folder_path = "Plot_Results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = "Pareto_SPD_" + '_'+ dataset+'_' +distribution+'_'+ mdl+'_'+ method+  "_DATASET.pdf"
    file_path = os.path.join(folder_path, filename)
    plt.savefig(file_path)
    
    plt.show()
    
def plot_fairness_weight_Pareto_EOD(eod, acc, acc_std, eod_std, fairness_weights, dataset,distribution,mdl,method):
    # Define markers and colors for each fairness weight (customize as needed)
    markers = ['o', 's', 'D', '^', 'v', 'p', '*']  # Added 'p' for pentagon and '*' for star
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Added 'y' for yellow and 'k' for black


    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    for i, fairness_weight in enumerate(fairness_weights):
        fairness_weight = str(fairness_weight)
        plt.errorbar(eod[fairness_weight], acc[fairness_weight], xerr=eod_std[fairness_weight], yerr=acc_std[fairness_weight], 
             marker=markers[i], color=colors[i], label=f'Fairness Weight {float(fairness_weight):.4f}', linestyle='none', markersize=8)



    # Label the axes
    plt.xlabel('Equalized Odds (EOD)')
    plt.ylabel('Accuracy (ACC)')

    # Add a legend
    plt.legend()

    # Set title and show the plot
    plt.title('Pareto_EOD_'+ '_'+ dataset+'_' +distribution+'_'+ mdl+'_'+ method+ '_DATASET')
    plt.grid(True)
    
    ## Saving Fig
    folder_path = "Plot_Results"
    filename = "Pareto_EOD_" + '_'+ dataset+'_' +distribution+'_'+ mdl+'_'+ method+  "_DATASET.pdf"
    file_path = f"{folder_path}/{filename}"
    plt.savefig(file_path)
    
    plt.show()
    
def plot_convergent_Pareto_SPD(spd, acc, acc_std, spd_std,dataset,distribution,mdl):
    # Extract method names from the keys of the SPD dictionary
    methods = list(spd.keys())

    # Define markers and colors for each method (customize as needed)
    markers = ['o', 's', 'D', '^', 'v']
    colors = ['b', 'g', 'r', 'c', 'm']

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    for i, method in enumerate(methods):
        plt.errorbar(spd[method], acc[method], xerr=spd_std[method], yerr=acc_std[method], 
                     marker=markers[i], color=colors[i], label=method, linestyle='none', markersize=8)

    # Label the axes
    plt.xlabel('Statistical Parity(SPD)')
    plt.ylabel('Accuracy (ACC)')

    # Add a legend
    plt.legend('lower right')

    # Set title and show the plot
    plt.title('SPD vs ACC_'+dataset+'_'+ distribution+ mdl + '_DATASET')
    plt.grid(True)
    
    ## Saving Fig
    folder_path = "Plot_Results"
    filename = "SPD vs ACC_" +dataset+'_'+ distribution+ mdl + "_DATASET.pdf"
    file_path = f"{folder_path}/{filename}"
    plt.savefig(file_path)
    
    
    plt.show()


def plot_convergent_Pareto_EOD(eod, acc, acc_std, eod_std,dataset,distribution,mdl):
    # Extract method names from the keys of the EOD dictionary
    methods = list(eod.keys())

    # Define markers and colors for each method (customize as needed)
    markers = ['o', 's', 'D', '^', 'v']
    colors = ['b', 'g', 'r', 'c', 'm']

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    for i, method in enumerate(methods):
        plt.errorbar(eod[method], acc[method], xerr=eod_std[method], yerr=acc_std[method], 
                     marker=markers[i], color=colors[i], label=method, linestyle='none', markersize=8)

    # Label the axes
    plt.xlabel('Equalized Odds (EOD)')
    plt.ylabel('Accuracy (ACC)')

    # Add a legend
    plt.legend('lower right')

    # Set title and show the plot
    plt.title('EOD vs ACC_'+dataset+'_'+ distribution+ mdl+'_DATASET')
    plt.grid(True)
    
    ## Saving Figure
    folder_path = "Plot_Results"
    filename = "EOD vs ACC_" +dataset+'_'+ distribution+mdl+ "_DATASET.pdf"
    file_path = f"{folder_path}/{filename}"
    plt.savefig(file_path)
    
    plt.show()

    

############ Graph Plotting ####################33
#dataset = ['COMPAS']
model = ['LR']  # 'NN' --> KRTD and KRTWD
# methods = ['KRTWD', 'KRTD', 'FedAvg','MinMax','FairFed_w_FairBatch']
dist = ['Non-IID']  # 'Non-IID' --> MinMax (one client does not have the samples) and FairFed_w_FairBatch


# fairness_weight = 1e1


# #########################    Doing Individual Plots 
# for m in methods:
#     for d in dataset:
#         for mdl in model:
#             for distribution in dist: 
#                   if m == 'KRTWD' or  m =='KRTD': 
#                       filename = f"{m}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results_90_10.pickle"
#                   else: 
#                       filename = f"{m}_{mdl}_{d}_{distribution}_test_results_90_10.pickle" 
#                   plot_individual(filename)


##############   Convergent SPD vs Acc for one choice ###############################
# import numpy as np

# acc = {}
# spd = {}
# eod = {}
# acc_std = {}
# eod_std = {}
# spd_std = {}

# fairness_weights = list( np.logspace(-3, -1, 7))
# #fairness_weights = fairness_weights[:-1] 
# model = ['LR','NN']  # 'NN' --> KRTD and KRTWD
# #methods = ['FairFed_w_FairBatch', 'FedAvg','MinMax']
# methods = ['KRTD','KRTWD'] #'KRTD']
# dist = ['IID','Non-IID']#'Non-IID']
# dataset = ['COMPAS'] #'COMPAS']

# for m in methods:
#     for d in dataset:
#         for mdl in model:
#             for distribution in dist: 
#                 for fairness_weight in fairness_weights:
#                     #import pdb;pdb.set_trace()
#                     if m == 'KRTWD' or  m =='KRTD':
#                         if(distribution =='Non-IID'):
#                             filename = f"{m}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results_90_10.pickle"
#                         else:
#                             filename = f"{m}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results.pickle"
#                     else:
                        
#                         if(distribution =='Non-IID'):
#                             # print("Here")
#                             filename = f"{m}_{mdl}_{d}_{distribution}_test_results_90_10.pickle" 
#                         else:
#                             filename = f"{m}_{mdl}_{d}_{distribution}_test_results.pickle"
                        
#                 acc_temp,spd_temp,eod_temp, acc_std_temp, spd_std_temp, eod_std_temp = get_conv_acc_spd(filename)
#                 acc[m] = acc_temp
#                 spd[m] = spd_temp
#                 eod[m] = eod_temp
#                 acc_std[m] = acc_std_temp
#                 eod_std[m] = eod_std_temp
#                 spd_std[m] = spd_std_temp

#                 plot_convergent_Pareto_EOD(eod,acc,acc_std,eod_std,d,distribution,mdl)

#                 plot_convergent_Pareto_SPD(spd,acc,acc_std,spd_std,d,distribution,mdl)


######################### Plot the entire Pareto ################
import numpy as np
acc = {}
spd = {}
eod = {}
acc_std = {}
eod_std = {}
spd_std = {}
dataset = ['COMPAS']
commcost = {}
fairness_weights = list( np.logspace(-3, -1, 7))
methods = ['KRTD','KRTWD'] #'KRTD' ] #'FedAvg','MinMax','FairFed_w_FairBatch']
model = ['LR','NN']
dist = ['IID','Non-IID'] 



for method in methods:
    for d in dataset:
        for mdl in model:
            for distribution in dist:
                for fairness_weight in fairness_weights:
                    
                    if method == 'KRTWD' or 'KRTD':
                        if(distribution =='Non-IID'):
                            filename = f"{method}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results_90_10.pickle"
                        else:
                            filename = f"{method}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results.pickle"
                        # Call the get_conv_acc_spd function to obtain results
                        acc_temp, spd_temp, eod_temp, acc_std_temp, spd_std_temp, eod_std_temp = get_conv_acc_spd(filename)
                        fairness_weight = str(fairness_weight)
                        
                        # Store results in dictionaries using fairness_weight as the key
                        acc[fairness_weight] = acc_temp
                        spd[fairness_weight] = spd_temp
                        eod[fairness_weight] = eod_temp
                        acc_std[fairness_weight] = acc_std_temp
                        eod_std[fairness_weight] = eod_std_temp
                        spd_std[fairness_weight] = spd_std_temp
                    else:
                        filename = f"{method}_{mdl}_{d}_{distribution}_test_results.pickle"

                    # Get the communication cost
                    commcost[method] = get_all_comcost(filename)

                    
                plot_fairness_weight_Pareto_SPD(spd, acc, acc_std, spd_std, fairness_weights,d,distribution,mdl,method)

                plot_fairness_weight_Pareto_EOD(eod, acc, acc_std, eod_std, fairness_weights, d,distribution,mdl,method)

################## Getting all the comm_cost ################################


# print(commcost['KRTD'])

# print(commcost['KRTWD'])


# print(commcost['MinMax'])

# print(commcost['FairFed_w_FairBatch'])