import matplotlib.pyplot as plt
import pickle


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
    plt.legend()

    # Set title and show the plot
    plt.title('SPD vs ACC_'+dataset+'_'+ distribution+ mdl + '_DATASET')
    plt.grid(True)
    
    ## Saving Fig
    folder_path = "Plot_Results"
    filename = "SPD vs ACC_" +dataset+'_'+ distribution+ mdl + "_DATASET.pdf"
    file_path = f"{folder_path}/{filename}"
    plt.savefig(file_path)
    
    
    plt.show()