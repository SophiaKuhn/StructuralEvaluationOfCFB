# Author(s): Sophia Kuhn (ETH Zürich)

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import math as m
import os
from skopt.sampler import Lhs, Sobol
import pandas as pd


#parameter specific functions
def ecu_calc(fccs): # EN 1992-1-1:2004 (Table 3.1)
    # Attention! this is calculated with fck? check in Euro code. And we input fcd!!
    return fccs.apply(lambda fcc: -0.002 if fcc <= 50 else ((2.0+0.085*(fcc-50)**(0.53)) *-0.001))


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
        #axes[i].set_title(f'Histogram of {column}')
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
#concept idea:
#TODO
# have libary of parameter sample spaces here (for documentation purposes, better that in the notebook I think)
# 

parametersStudyLib={'1.0':
                        {'description':'Defined on 22.05.2024. First high-dimensional parameter study. Meshsize 3, homogenious reinforcement in plate and wall respectively, no voute, no skew bridges, no sections, no parapet ',
                        'ranges' :{
                                    #geometry parameter
                                    'L': (2000.,18000.), #span [mm]
                                    'b1' : (3000.,20000.), # witdh of wall(s) [mm]
                                    't_p': (200.,1200.), # Deck slab (plate) thickness [mm]
                                    't_w':(200., 1200.),   # wall thickness [mm]
                                    'h_w' :(2000.,5000.), #wall height [mm]
                                    # reinforcement parameter
                                    'd1_plate' :(10.,30.), # Reinforcement Diameter of all plate sections - Layer 1 (lower outer layer) [mm]
                                    'd4_plate' :(10.,30.), #Reinforcement Diameter of all plate sections - Layer 4 (upper outer layer) [mm]
                                    'd2_plate' :(10.,30.), # Reinforcement Diameter of all plate sections - Layer 2 (lower middle layer) [mm]
                                    's_plate' :(100.,500.), # Reinforcement spacing of all plate sections [mm]
                                    'd1_walls' :(10.,30.), # Reinforcement Diameter of all plate sections - Layer 1 (inner layer) [mm]
                                    'd4_walls' :(10.,30.), # Reinforcement Diameter of all walls sections - Layer 4 (outer layer) [mm]
                                    's_walls' :(100.,500.), # Reinforcement spacing of all walls sections [mm]
                                    # material parameter
                                    'fcc' :(25.,50.), # Concrete compressive strength
                                    # loading parameter
                                    's' :(0.1,0.9), #Distance factor between origin and track middle axis [-]
                                    'beta' :(-50.,50.), #Angle between global x axis and track axis [Degree]
                                    'h_G' :(300.,1500.), #Gravel layer hight [mm]
                                    },

                        'constants' : {
                                    #geometry parameter
                                    #'b2': b1, #Width of wall 2 is kept equal to width of wall 1 for this study
                                    'alpha_l': [90], # Plate Angle [Degree]
                                    'h_v': [0] , # Hight of Voute  [mm]
                                    'l_v': [0], # Length of Voute  [mm]
                                    # reinforcement parameter
                                    'oo' : [30], #upper reinforcement cover [mm],
                                    'uu' : [30], #lower reinforcement cover [mm]
                                    'd3_plate': [12], # Reinforcement Diameter of all plate sections - Layer 3 (upper middle layer) [mm]
                                    'd3_walls': [12], # Reinforcement Diameter of all walls sections - Layer 3 (outer middle layer) [mm]
                                    'd2_walls': [12], # Reinforcement Diameter of all walls sections - Layer 2 (inner middle layer) [mm]

                                    # material parameter
                                    'fsy': [390], # Reinforcement steel yield strength
                                    'fsu_fac': [1.08], # Reinforcement steel ultimate strength factor
                                    'esu': [0.045], # ultimate reinforcement strain [-]
                                    'ecu' : [-0.002], # ultimate concrete strain [-]

                                    # loading parameter
                                    'gamma_E':[0.00002], #spez. weight of backfill/gravel [N/mm3]
                                    'phi_k': [30], # friction angle of backfill/gravel [Degree]
                                    'q_Gl': [4.8+1.7], # Load of concrete sleeper (Betonschwelle) and rail track [N/mm]
                                    'b_Bs':[2500] , # width of concrete sleeper (Betonschwelle) [mm]
                                    'Q_k':[225000], # norminal axle load (dependent on the class, acc. to SIA 269/1 11.2, for D4 = 225 [N])

                                    #section geom. parameter                    
                                    'l_sec': [0.33] , # Proportion of outter plate sections [-]
                                    'b_sec1_b1': [0.33], # Proportion of section 1 (wall 1 side) [-]
                                    'b_sec2_b1': [0.33], # Proportion of section 2 (wall 1 side) [-]
                                    # 'b_sec1_b2': [0.33], # Proportion of section 1 (wall 2 side) [-]
                                    # 'b_sec2_b2': [0.33], # Proportion of section 2 (wall 2 side) [-]
                                    'h_S1_3': [0.33], # Proportion of wall section (upper)
                                    'h_S7_9': [0.33], # Proportion of wall section (lower)

                                    #modelling parameter
                                    'mesh_size_factor': [3], # mesh_size_factor (multiplied with minimum of t_p  and t_w)
                                    }
                        },

                    '1.1':
                        {'description':None,
                        'ranges': None,
                        'constants' : None}
                    
                        

                    }


    

class CFBSamplingSpace:

    """
    Sampling space class for concret frame bridges (CFB) with built in sampler functions.
    In combination with parametersStudyLib, which is simply a dict where all parameter studies are saved.

    Parameters
    ----------
    parameterStudie: str
        Name of parameter Study that we want to get from parametersStudyLib
    """

    def __init__(self, parameterStudy):
        
        self.parameterStudie=parameterStudy

        if parameterStudy in parametersStudyLib.keys():
            prameterStudyDict=parametersStudyLib[parameterStudy]
        else:
            raise Exception('The parameter study name you provided is not in the parameter study library. See sampling.py script.')
        
        self.description=prameterStudyDict['description']
        self.ranges=prameterStudyDict['ranges']
        self.constants=prameterStudyDict['constants']

        

    def LHS(self,n_samples):
        """
        Perfromed Latin Hyper Cube Sampling with defined sampling space.
        So it samples the variable parameters within the defined ranges and then adds the constant parameters to each sample.

        Parameters
        ----------
        n_samples: int
            Number of samples to sample


        Returns
        ----------
        df_x: DataFrame
            A pandas DataFrame with the sampled parameters (incl. the constants).
        """

        # Sample the variable parameter
        ranges_list = list(self.ranges.values())
        variable_names= list(self.ranges.keys())

        lhs=Lhs(lhs_type='classic', criterion='maximin', iterations=1000)
        X=lhs.generate(ranges_list, n_samples, random_state=None)

        # tranform into datafram
        df_x=pd.DataFrame(X)
        df_x.columns =  variable_names # renaming columns


        #add constant parameters
        for key in self.constants.keys():
            df_x[key]=self.constants[key][0]

        return df_x
            




    
    
        # # if specific parameter stu
        # if parameterStudie==1:
        #     self.ranges={k: parameterStudie1['Ranges'][k] for k in variables if k in parameterStudie1['Ranges']}
        #     self.constants=parameterStudie1['Constants']

        
###### OLD CODE From here onwards to deleate or use

# hero = {'L': 5800,
#         'b1': 9720,
#         't_p': 400,
#         't_w': 400,
#         'h_w': 2580,
#         'd1_plate': 24,
#         'd4_plate': 24,
#         'd2_plate':12,
#         's_plate': 200,
#         'd1_walls': 14,
#         'd4_walls': 24,
#         's_walls': 200,
#         'fcc': 12.8,
#         'fsy': 390,
#         'fsu_fac': 1.08,
#         's': 0.18179012,
#         'beta': 5}
        

    
    # def _get_parameterStudie(self):

    #     parameterStudie=self.parameterStudie


    #     if parameterStudie == 1:

    #         variableRanges ={'L': (2000.,18000.),
    #                          'b1' : (3000.,20000.)
    #                          }
            
    #         variableConstants ={'fsy': 390,
    #                             'oo' : 30,
    #                             'd3_plate': 12
    #                             }
        

    #     else:
    #         raise Exception('The defined parameterStudie is invalid.')
        
        
    #     return variableRanges,variableConstants

        


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
