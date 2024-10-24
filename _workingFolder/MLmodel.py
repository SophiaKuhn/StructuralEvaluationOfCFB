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



#-----------------------------------------------------------------------------------------------
# Model Setup
#-----------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------
# Training
#-----------------------------------------------------------------------------------------------





#-----------------------------------------------------------------------------------------------
# Prediction
#-----------------------------------------------------------------------------------------------

def bnn_predict_with_uncertainty(model, x, n=1000, log_transform_back=False):
    """
    Generate n predictions for each sample in x_data using the Bayesian neural network model.
    
    Args:
    - model: The trained Bayesian neural network model.
    - x: Input data (torch tensor) for which predictions are to be made. Attention: x has to be transformed like the x data the model was trained with!
    - n: Number of predictions to generate for each sample.
    - log_transform_back: Set to true of predicted values are scaled with ln(y+1) and have to be retransformed (bool).
    
    Returns:
    - y_pred_mean_values: Mean predictions for each sample (numpy array).
    - y_pred_std_values: Standard deviation of predictions for each sample (numpy array).
    """
    # Generate n predictions for the input data
    y_pred_n_times = np.array([model(x).data.numpy() for k in range(n)])

    # Reshape and transpose the results for easier computation
    y_pred_n_times = y_pred_n_times[:, :, 0] # each row corresponds to the nth prediction for each data point
    y_pred_n_times = y_pred_n_times.T # each row corresponds to the predictions for a single data point across all n iterations
    
    # Transform back from log scale
    if log_transform_back:
        y_pred_n_times = np.exp(y_pred_n_times) - 1

    # Calculate mean and standard deviation of predictions for each sample
    y_pred_mean_values = np.array([y_pred_n_times[i].mean() for i in range(len(y_pred_n_times))]).reshape(-1, 1)
    y_pred_std_values = np.array([y_pred_n_times[i].std() for i in range(len(y_pred_n_times))]).reshape(-1, 1)

    return y_pred_mean_values, y_pred_std_values, y_pred_n_times


#-----------------------------------------------------------------------------------------------
# Metrics
#-----------------------------------------------------------------------------------------------

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true values and mean predicted values.

    Args:
    - y_true: True values (numpy array).
    - mean_values: Mean predicted values (numpy array).

    Returns:
    - rmse: The calculated RMSE value (float).
    """
    # check if correct type of input
    if isinstance(y_true, torch.Tensor):
        raise Exception('y_true is a torch tensor. It should be of the type numpy array.')

    if isinstance(y_pred, torch.Tensor):
        raise Exception('y_pred is a torch tensor. It should be of the type numpy array.')

    #check correct format of input
    # Check that y_true and y_pred have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f'Shape mismatch: y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}.')


    # Calculate the squared differences
    squared_diff = (y_true - y_pred) ** 2

    # Calculate the mean of squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_diff)

    return rmse


#-----------------------------------------------------------------------------------------------
# Uncertanty Callibration
#-----------------------------------------------------------------------------------------------


def calc_confinence_interval(confidence_level, mean_prediction, std_prediction):

    confidence_interval=np.array([[meanPred - z * std, meanPred + z * std]])






# ## Uncertanty calibaration plot ### Implemented acc to Hands on paper supplementry material space learner
# from scipy.stats import chi2 as Chi2Dist

# # calculate covariance
# deviations = y_train_pred_n_times - y_train_mean_pred_np # deviations of each stochastic prediction from the mean prediction (y_single pred - mean prediction)
# n_samples, n_predictions = y_train_pred_n_times.shape # get number of samples and number of predictions per sample

# cov_vector = np.zeros(n_samples) # initialize covariance matrix # for a single prediction this is a vector of length of number of sample

# # calculate covariance value for each sample individually and then put at right position in initialized matrix
# for i in range(n_samples):
#     deviation=deviations[i]
#     individual_preds = y_train_pred_n_times[i]
#     mean_pred = y_train_mean_pred_np[i]
#     cov=np.dot(deviations[i], deviations[i].T)/(n_predictions - 1) # as we only have one single output this is just a vector #same to np.var(deviations[i], axis=0, ddof=1)
#     cov_vector[i] = cov

# inv_cov_vector=(1/cov_vector).reshape(-1,1) # the inverse of a vector is simply the each value ^-1 (so the reciprocal)

# nssr=errors*inv_cov_vector*errors
# nssr=np.sort(nssr.flatten())
# p_obs = np.linspace(1./nssr.size,1.0,nssr.size)
# p_pred = Chi2Dist.cdf(nssr, 9)



# plt.figure()
# plt.plot(p_pred, p_obs, label='Model calibration curve')
# plt.plot([0,1],[0,1], 'k--', alpha=0.6, label='Ideal calibration curve')
# plt.xlabel('Predicted probability')
# plt.ylabel('Observed probability')
# plt.axis('equal')
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.legend()

# plt.show()

#-----------------------------------------------------------------------------------------------
# Utils
#-----------------------------------------------------------------------------------------------

def filter_values_within_range(y_true, y_pred, lb, ub):
    """
    Filter the true values and predicted values within a specified range.
    All samples that fall into the range (lower and upper bound) is identified on y_true, and then the same samples are also selected for y_pred.

    Args:
    - y_true: True values (numpy array).
    - y_pred: Predicted mean values (numpy array).
    - lb: Lower bound of the range.
    - ub: Upper bound of the range.

    Returns:
    - filtered_y_true: Filtered true values within the specified range (numpy array).
    - filtered_y_pred: Filtered predicted values within the specified range (numpy array).
    """

    # check if correct type of input
    if isinstance(y_true, torch.Tensor):
        raise Exception('y_true is a torch tensor. It should be a numpy array.')
        # check if correct type of input
    if isinstance(y_pred, torch.Tensor):
        raise Exception('y_pred is a torch tensor. It should be a numpy array.')
    
    #check correct format of input
    # Check that y_true and y_pred have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f'Shape mismatch: y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}.')


    # Create a mask for values within the specified range
    mask = (y_true >= lb) & (y_true <= ub) 

    # Apply the mask (and shape)
    filtered_y_true = y_true[mask].reshape(-1, 1)
    filtered_y_pred= y_pred[mask].reshape(-1, 1)


    return filtered_y_true, filtered_y_pred








########################### OLD Function to be aufgeräumt ######################3



# #data transformations
# # TODO add min may scaling here



# # loss functions
# def RMSE(y_pred, y_true, y_range=None, range_setting=('>=', '<='), verbalize=True):
#     """
#     ....description

#     Parameters
#     ----------
#     y_pred : trch.tensor
#         2 dimensional tensor of the predicted y values tensor([[a] [b]...])
#     y_true : trch.tensor
#         2 dimensional tensor of the true y values tensor([[a] [b]...])
#     verbalize: Boolean
#         weather the function should print
#     Returns
#     ----------
#     rmse : torch.tensor
#         torch sensor which includs the calculated rmse value

#     """
#     if not y_range==None:
#         print('A range is provided.')
#         lower_bound = y_range[0]
#         upper_bound = y_range[1]

#         # Create a mask for values within the specified range
#         if range_setting== ('>=', '<='):
#             mask = (y_true >= lower_bound) & (y_true <= upper_bound)
#         if range_setting== ('>', '<'):
#             mask = (y_true > lower_bound) & (y_true < upper_bound)
#         else:
#             NotImplementedError
        
#         y_true = y_true[mask]
#         y_pred = y_pred[mask]
        


#     squared_diff = (y_true - y_pred) ** 2
#     mse = torch.mean(squared_diff)
#     rmse = torch.sqrt(mse)
#     print("Root Mean Squared Error (RMSE):", rmse.item())

#     return rmse


# def predict(model, x_scaled, n=1000):


#     models_result = np.array([model(x_scaled).data.numpy() for k in range(n)])
#     models_result = models_result[:,:,0]    
#     models_result = models_result.T #transpose--> each row corresponds to the predictions for a single data point across all 10000 iterations
#     #get mean predictions form each sample
#     mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
#     #calculate standard derivation of prediction for each sample
#     std_values = np.array([models_result[i].std() for i in range(len(models_result))])

#     return mean_values, std_values

# # BNN class
# class BayesianNN:

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

#     def __init__(self,variables=None, parameterStudie=1):
        
#         self.parameterStudie=parameterStudie
        
#         if variables == None:
#             if parameterStudie==1:
#                 variables=parameterStudie1['Ranges']

#             else:
#                 raise Exception('Not Implemented')
#         self.variables=variables


#         if parameterStudie==1:
#             self.ranges={k: parameterStudie1['Ranges'][k] for k in variables if k in parameterStudie1['Ranges']}
#             self.constants=parameterStudie1['Constants']

        



        

    
#     def _get_parameterStudie(self):

#         parameterStudie=self.parameterStudie


#         if parameterStudie == 1:

#             variableRanges ={'L': (2000.,18000.),
#                              'b1' : (3000.,20000.)
#                              }
            
#             variableConstants ={'fsy': 390,
#                                 'oo' : 30,
#                                 'd3_plate': 12
#                                 }
        

#         else:
#             raise Exception('The defined parameterStudie is invalid.')
        
        
#         return variableRanges,variableConstants

        


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
