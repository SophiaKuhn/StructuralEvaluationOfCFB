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