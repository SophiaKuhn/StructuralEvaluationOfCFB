# Author(s): Sophia Kuhn (ETH Zürich)

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import math as m
import os


# util function
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{folder_path}': {e}")

def create_subfolders_for_samples(idx_s, n_samples, folder_path):

    #create data folder if does not exists yet
    create_folder_if_not_exists(folder_path)

    #TODO deleate all subfolders in that folder

    # check/ create batch folder
    subfolder = '{}_Batch'.format(idx_s)
    subfolder_path = folder_path+ '\\' +subfolder
    create_folder_if_not_exists(subfolder_path)

    # check/ create subfolders for each bridge
    for i in range(n_samples):
        indiv_path= subfolder_path + '\\{}_{}_CFB'.format(idx_s,i)
        create_folder_if_not_exists(indiv_path)



#plotting function
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
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(False)  # Optional: Remove grid lines

    # Hide any empty subplots if the number of columns is less than n_rows*n_cols
    for ax in axes[len(df.columns):]:
        ax.axis('off')

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()  # Display the histograms


### from here downward code in progress...!
#sampler
    
parameterStudie1={'Ranges' :{'L': (2000.,18000.),
                                    'b1' : (3000.,20000.),
                                    't_p': (200.,1200.),

                                    't_w':(200., 1200.),   # Wall Thickness [mm]
                                    'h_w' :(2000.,5000.), 
                                    },
                'Constants' : {'fsy': 390,
                                        'oo' : 30,
                                        'd3_plate': 12
                                    }
                }

hero = {'L': 5800,
        'b1': 9720,
        't_p': 400,
        't_w': 400,
        'h_w': 2580,
        'd1_plate': 24,
        'd4_plate': 24,
        'd2_plate':12,
        's_plate': 200,
        'd1_walls': 14,
        'd4_walls': 24,
        's_walls': 200,
        'fcc': 12.8,
        'fsy': 390,
        'fsu_fac': 1.08,
        's': 0.18179012,
        'beta': 5}


    
variableRanges ={'L': (2000.,18000.),
                             'b1' : (3000.,20000.)
                             }
    

class CFBSamplingSpace:

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
