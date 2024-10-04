# Author(s): Sophia Kuhn (ETH Zürich)

import pandas as pd
import math as m
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

from sklearn.neighbors import KernelDensity


#-----------------------------------------------------------------------------------------------
# Data Distributions
#-----------------------------------------------------------------------------------------------


def hist_matrix(df, n_cols=4, bins=20, color='gray', edgecolor='darkgray'):
    # Number of columns in the DataFrame
    n_columns = len(df.columns)

    # Number of histogram columns per row
    n_cols = 4

    # Calculate the number of rows needed
    n_rows = m.ceil(n_columns / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 2.5))  # Adjust the figsize as needed
    axes = axes.flatten()

    for i, column in enumerate(df.columns):
        # Plot histogram on the corresponding subplot
        axes[i].hist(df[column], bins=bins, color=color, edgecolor=edgecolor)  # You can customize the histogram here
        #axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(False)  # Optional: Remove grid lines

    # Hide any empty subplots if the number of columns is less than n_rows*n_cols
    for ax in axes[len(df.columns):]:
        ax.axis('off')

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()  # Display the histograms




def hist_kde_matrix(df, bandwidth=1, n_cols=4, bins=20, color='gray', edgecolor='darkgray', pde_color='navy'): #dodgerblue
    # Number of columns in the DataFrame
    n_columns = len(df.columns)

    # Number of histogram columns per row
    n_cols = 4

    # Calculate the number of rows needed
    n_rows = m.ceil(n_columns / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 2.5))  # Adjust the figsize as needed
    axes = axes.flatten()

    for i, column in enumerate(df.columns):

        # fit 1d kde
        data=df[column].values.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)
        # Create a grid of values (covering the range of your data)
        x_grid = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)
        # Evaluate the PDF on the grid
        log_pdf = kde.score_samples(x_grid)
        pdf = np.exp(log_pdf)

        # Plot histogram on the corresponding subplot
        # Plot histogram and keep a patch for legend
        n, bins, patches = axes[i].hist(df[column], bins=bins, density=True, color=color, edgecolor=edgecolor, alpha=0.7)
        if i == 0:  # Save the patch for the legend
            hist_patch = patches[0]

        # Plot KDE and keep a line for legend
        kde_line, = axes[i].plot(x_grid, pdf, color=pde_color, lw=2)  # Keep the line object for legend
        
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Density')
        axes[i].grid(False)  # Optional: Remove grid lines

    # Hide any empty subplots if the number of columns is less than n_rows*n_cols
    for ax in axes[len(df.columns):]:
        ax.axis('off')
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])


        # Add a legend to the figure
    fig.legend([hist_patch, kde_line], ['Histogram of the data', 'KDE fitted PDE'], loc='upper left', bbox_to_anchor=(1, 0.9))


    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()  # Display the histograms

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
            plt.savefig(f"{save_path}/{model_name}_{name}_loss_plot.png")
        plt.show()



#-----------------------------------------------------------------------------------------------
# Model Evaluation Plots
#-----------------------------------------------------------------------------------------------

def plot_true_vs_pred(y_true, y_pred, rmse_value, y_name, title=None, xlim=None,ylim=None ):
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


    if not ylim==None:
        plt.ylim((ylim[0],ylim[1]))
    if not xlim==None:
        plt.ylim((xlim[0],xlim[1]))
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