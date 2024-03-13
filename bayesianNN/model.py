# Author(s): Sophia Kuhn (ETH Zürich)

from typing import Dict, List, Union

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt


#data transformations
# TODO add min may scaling here



# loss functions
def RMSE(y_pred, y_true, y_range=None, range_setting=('>=', '<='), verbalize=True):
    """
    ....description

    Parameters
    ----------
    y_pred : trch.tensor
        2 dimensional tensor of the predicted y values tensor([[a] [b]...])
    y_true : trch.tensor
        2 dimensional tensor of the true y values tensor([[a] [b]...])
    verbalize: Boolean
        weather the function should print
    Returns
    ----------
    rmse : torch.tensor
        torch sensor which includs the calculated rmse value

    """
    if not y_range==None:
        print('A range is provided.')
        lower_bound = y_range[0]
        upper_bound = y_range[1]

        # Create a mask for values within the specified range
        if range_setting== ('>=', '<='):
            mask = (y_true >= lower_bound) & (y_true <= upper_bound)
        if range_setting== ('>', '<'):
            mask = (y_true > lower_bound) & (y_true < upper_bound)
        else:
            NotImplementedError
        
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        


    squared_diff = (y_true - y_pred) ** 2
    mse = torch.mean(squared_diff)
    rmse = torch.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse.item())

    return rmse


def predict(model, x_scaled, n=1000):


    models_result = np.array([model(x_scaled).data.numpy() for k in range(n)])
    models_result = models_result[:,:,0]    
    models_result = models_result.T #transpose--> each row corresponds to the predictions for a single data point across all 10000 iterations
    #get mean predictions form each sample
    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    #calculate standard derivation of prediction for each sample
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])

    return mean_values, std_values

# BNN class
class BayesianNN:

    """
    Samples values according to certain strategies.

    Parameters
    ----------
    strategies : List[Strategy]
        List of strategies to be used for sampling.
    objective : Operator, optional, default=None
        Objective to be optimised. The sampler is trained using the objective values of the samples, in order
        to optimize future sampling campaigns,
    """

    def __init__(self,variables=None, parameterStudie=1):
        
        self.parameterStudie=parameterStudie
        
        if variables == None:
            if parameterStudie==1:
                variables=parameterStudie1['Ranges']

            else:
                raise Exception('Not Implemented')
        self.variables=variables


        if parameterStudie==1:
            self.ranges={k: parameterStudie1['Ranges'][k] for k in variables if k in parameterStudie1['Ranges']}
            self.constants=parameterStudie1['Constants']

        



        

    
    def _get_parameterStudie(self):

        parameterStudie=self.parameterStudie


        if parameterStudie == 1:

            variableRanges ={'L': (2000.,18000.),
                             'b1' : (3000.,20000.)
                             }
            
            variableConstants ={'fsy': 390,
                                'oo' : 30,
                                'd3_plate': 12
                                }
        

        else:
            raise Exception('The defined parameterStudie is invalid.')
        
        
        return variableRanges,variableConstants

        


# class SamplesGenerator:



#     """
#     Samples values according to certain strategies.

#     Parameters
#     ----------
#     strategies : List[Strategy]
#         List of strategies to be used for sampling.
#     objective : Operator, optional, default=None
#         Objective to be optimised. The sampler is trained using the objective values of the samples, in order
#         to optimize future sampling campaigns,
#     """

#     def __init__(self,n_samples):
#         self.verbalise='Test'

# ==============================================================================
# Debugging
# ==============================================================================


if __name__ == "__main__":

    pass
