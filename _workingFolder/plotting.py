# Author(s): Sophia Kuhn (ETH Zürich)

import pandas as pd
import math as m
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn



#-----------------------------------------------------------------------------------------------
# Loss Plots 
#-----------------------------------------------------------------------------------------------

def plot_loss_development(loss_lists, loss_names, save_path=None, model_name=None, color='navy', figsize=None):
    """
    Plot the development of different loss functions during training.

    Args:
    - loss_lists: List of lists, where each list contains the loss values for a specific loss function.
    - loss_names: List of strings, where each string is the name of the corresponding loss function.
    - save_path: Optional path to save the plots. If None, plots are not saved.
    """
    if len(loss_lists) != len(loss_names):
        raise ValueError("The number of loss lists must match the number of loss names.")
    
    # Plot each loss function in separate plots
    for loss, name in zip(loss_lists, loss_names):
        if figsize==None:
            plt.figure(figsize=(10, 6))
        else: 
            plt.figure(figsize=figsize)
        plt.plot(loss, label=name, lw=2, color=color)
        plt.title(f'{name} Development During Training')
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(f"{save_path}/{name}_loss_plot{model_name}.png")
        plt.show()



#-----------------------------------------------------------------------------------------------
# Model Evaluation Plots
#-----------------------------------------------------------------------------------------------

def plot_true_vs_pred(y_true, y_pred, rmse_value, y_name, title=None):
    """
    Plot the true values vs. predicted values and the RMSE.

    Args:
    - y_true: True values (numpy array).
    - y_pred: Predicted mean values (numpy array).
    - rmse_value: The calculated RMSE value.
    - xlabel: Label for the x-axis (default is 'True $\eta_{min},_{concrete}$').
    - ylabel: Label for the y-axis (default is 'Predicted $\eta_{min},_{concrete}$').
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
    
    # plot
    plt.figure(figsize=(5, 4))
    plt.plot(y_true, y_pred, '.', color='gray', lw=3, label='Mean Predictions (RMSE: {:.2f})'.format(rmse_value))
    plt.grid(color='gray', linestyle='-', linewidth=0.5)

    # Determine the range for the x=y line
    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], 'k--', label='Perfect Prediction')

    plt.legend()
    xlabel='True '+y_name
    plt.xlabel(xlabel)
    ylabel='Predicted '+y_name
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()




########################### OLD Function to be aufgeräumt ######################3


# loss functions
def single_para_strudy_perf(x, y_pred, y_true, y_std, variable, unit=None, rmse=None, rmse_range=None, pred_color='darkgreen', true_color='black', ylim=None, xlim=None, figsize=(5,4),verbalize=True):
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

    plt.figure(figsize=figsize)

    #plot prediction model an uncertanty range
    plt.plot(x,y_pred, linestyle='--',color=pred_color ,lw=1.5,label='Predicted Mean Model')
    plt.plot(x,y_pred,'.', color=pred_color ,markersize=6,label='Predicted Mean')
    plt.fill_between(x.flatten(),y_pred.flatten()-3.0*y_std.flatten(),y_pred.flatten()+3.0*y_std.flatten(),alpha=0.2,color='darkgreen',label='99.7% confidence interval')
    
    #plot ground truth
    plt.plot(x,y_true,linestyle='--',color=true_color,lw=1.5)
    plt.plot(x,y_true,'x',color=true_color,markersize=6,label='NLFE-Analysis Results')

    # Add a red dotted line at y=1
    plt.axhline(y=1, color='r', linestyle=':', label='$\eta = 1$')

    # formatize
    if not ylim==None:
        plt.ylim((0,ylim))
    if not xlim==None:
        plt.ylim((0,xlim))
    plt.grid(color='gray', linestyle='-', linewidth=0.5) 
    plt.legend()
    plt.xlabel(variable+' '+unit)
    if not rmse==None:
        if not rmse_range==None:
            plt.title('total RMSE: {:.2f}, range RMSE: {:.2f}'.format(rmse,rmse_range))
        else:
            plt.title('total RMSE: {:.2f}'.format(rmse))
    plt.ylabel('$\eta_{c} min$')