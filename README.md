## KFFL
This repository contains code for Fair Federated Learning


## Instructions for Running the code:

 To create the environment do the following : 

    conda env create -f environment.yml
    
    conda activate environment.yml 


The code is modular and the simplest way to run the code is to use the following command in the terminal

    - `python3 test_functions.py` . 

 contains the following parameters that can be changed:

 - ***Dataset***: (ADULT or COMPASS). 
 - ***Distribution :***  (IID, Non-IID).
 - ***Methods (Baselines):*** 1)KRTWD **(KFFL)**,  2) KRTD **(KFFL-TD)** , 3) MinMax,4) FairFed/FairBatch, 5) Fed-Avg, 6) Central **(Centralized_KHSIC)**
 - ***Number of Simulations*** : Adjust for inspecting statistical variance in the output
 -   **Fairness Weights**: Adjust for desired tradeoff


## Sample Output

Running the above file yields the following output 

    accuracy 0.8347153123272526: SPD: 0.1567 EOD: 0.1631 Cost:  0.0000

The `accuracy` is the test accuracy.  `SPD` and `EOD`  are the evaluation metrics corresponding to statistical parity and equalized odds. (Ignore the term `Cost` as that is not implemented correctly).

## Main Support Files (Explanation)

    main.py  
Gives you the control to choose **step size, batch size, local epochs** and **number of clients** in the federated learning process.

    methods.py
Has all the baseline methods  (KFFL and baselines) implemented.

    kernel_utils.py
Contains the implementation for helper functions, client and server functions,KHSIC etc.


