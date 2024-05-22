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
#concept idea:
#TODO
# have libary of parameter sample spaces here (for documentation purposes, better that in the notebook I think)
# 
    
parameterStudie1={'Ranges' :{
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
                            's' :(0.1,0.9),
                            'beta' :(-50,50),
                            'h_G' :(300,1500),
                            },

                'Constants' : {
                                #geometry parameter
                                #'b2': b1, #Width of wall 2 is kept equal to width of wall 1 for this study
                                'alpha_l': 90, # Plate Angle [Degree]
                                'h_v': [0] , # Hight of Voute  [mm]
                                'l_v': [0], # Length of Voute  [mm]
                                # reinforcement parameter
                                'oo' : 30, #upper reinforcement cover [mm],
                                'uu' : 30, #lower reinforcement cover [mm]
                                'd3_plate': 12, # Reinforcement Diameter of all plate sections - Layer 3 (upper middle layer) [mm]
                                'd3_walls': 12, # Reinforcement Diameter of all walls sections - Layer 3 (outer middle layer) [mm]
                                'd2_walls': 12, # Reinforcement Diameter of all walls sections - Layer 2 (inner middle layer) [mm]

                                # material parameter
                                'fsy': 390, # Reinforcement steel yield strength
                                'fsu_fac': 1.08, # Reinforcement steel ultimate strength factor
                                'esu': 0.045, # ultimate reinforcement strain [-]

                                # loading parameter
                                'gamma_E':0.00002, #spez. weight of backfill/gravel [N/mm3]
                                'phi_k': 30, # friction angle of backfill/gravel [Degree]
                                'q_Gl': 4.8+1.7, # Load of concrete sleeper (Betonschwelle) and rail track [N/mm]
                                'b_Bs':2500 , # width of concrete sleeper (Betonschwelle) [mm]
                                'Q_k':225000, # norminal axle load (dependent on the class, acc. to SIA 269/1 11.2, for D4 = 225 [N])

                                #section geom. parameter                    
                                'l_sec': [0.33 ], # Proportion of outter plate sections [-]
                                'b_sec1_b1': [0.33], # Proportion of section 1 (wall 1 side) [-]
                                'b_sec2_b1': [0.33], # Proportion of section 2 (wall 1 side) [-]
                                'h_S1_3': [0.33], # Proportion of wall section (upper)
                                'h_S7_9': [0.33], # Proportion of wall section (lower)

                                #modelling parameter
                                'mesh_size_factor': [3] # mesh_size_factor (multiplied with minimum of t_p  and t_w)
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
        self.variables=variables
        
        # if only a parameter Study is selected
        if variables == None:
            if parameterStudie==1:
                variables=parameterStudie1['Ranges']

            else:
                raise NotImplementedError()
        

        else:
            raise NotImplementedError()
            
    
    
        # # if specific parameter stu
        # if parameterStudie==1:
        #     self.ranges={k: parameterStudie1['Ranges'][k] for k in variables if k in parameterStudie1['Ranges']}
        #     self.constants=parameterStudie1['Constants']

        



        

    
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
