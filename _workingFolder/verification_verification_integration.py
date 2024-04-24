 # Author(s): Sophia Kuhn (ETH Zurich)


import pandas as pd
import os
import json 
import pickle




def concrete_bending_verification(structure=None, results = None, step=None, return_type='dict', verbalise=True):

    '''
    This function evaluates the structure results with respect to concrete bending. 
    It caluclates eta based on the ration between the eps_c2d and the calculated eps. Eps here stands for strain.

    Parameters:
    structure: structure object
        a structure object defined after see function ....
    results: dict
        Contains the analysis results for each evaluation step.
    step: string
        the name of the analysis step (e.g. "step_4")
    return type: string
        the type of the return value. Possible return types are dict and df. The default is dict.
    verbalise: bool
        Flag that defines weather the function should print progress information (if set to True) 
        or should run without printing any progress information (if set to False).

    
    Returns:
    res: dict or pandas dataframe
        The function returns a dict with the verification results 
        The dict contains the following keys: 'eta_min_c', 'x_c', 'y_c', 'z_c', 'Location_c', 'GP_count_c'
    '''
    
    # if not results but the structure is given we first extract the results dict from the structure pkl file
    if results == None:
        if structure==None:
            raise Exception("You have not provided a structure of results. You have to provide at one!")
        
        results = structure.results
        if verbalise:
            print('The structure was converted to results dict')

    # save relevant values from dict into a data frame for easyer handling
    df_eps = pd.DataFrame({'eps_1_bot':list(results[step]['GP']['eps_1_bot'].values()),
                         'eps_3_bot':list(results[step]['GP']['eps_3_bot'].values()),
                         'eps_1_top':list(results[step]['GP']['eps_1_top'].values()),
                         'eps_3_top':list(results[step]['GP']['eps_3_top'].values()),
                         'coor_x_sig_sr_1L':list(results[step]['GP']['coor_x_sig_sr_1L'].values()),
                        'coor_y_sig_sr_1L':list(results[step]['GP']['coor_y_sig_sr_1L'].values()),
                         'coor_z_sig_sr_1L':list(results[step]['GP']['coor_z_sig_sr_1L'].values()),
                         } )       


    #define calculation of eta
    def calculate_eta_ifNeg(x):
        if x <0:
            #TODO: incase eps_c2d becomes a variable it has to be included here
            return -0.002/x
        else:
            return None

    # Calculate eta for for each GP an top and bot, for directions 1 and 2 (--> 4 times per GP) 
    df_eps['eta_1_bot']=df_eps['eps_1_bot'].apply(calculate_eta_ifNeg)
    df_eps['eta_3_bot']=df_eps['eps_3_bot'].apply(calculate_eta_ifNeg)
    df_eps['eta_1_top']=df_eps['eps_1_top'].apply(calculate_eta_ifNeg)
    df_eps['eta_3_top']=df_eps['eps_3_top'].apply(calculate_eta_ifNeg)

    # get minimum eta fpr each Gaus Point
    df_eps['eta_min_GP'] = df_eps[['eta_1_bot', 'eta_3_bot', 'eta_1_top', 'eta_3_top']].min(axis=1)
    # Get Minimum eta value of structure
    eta_min_structure = df_eps['eta_min_GP'].min()
    #min_value = df_eps[['eta_1_bot', 'eta_3_bot', 'eta_1_top', 'eta_3_top']].min().min()

    # Find the location (columnname) of the minimum eta value
    location = df_eps[['eta_1_bot', 'eta_3_bot', 'eta_1_top', 'eta_3_top']][df_eps == eta_min_structure].stack().index.tolist()[0][1]
    

    # Getting coordinates of min position
    idx_eta_min_structure = df_eps['eta_min_GP'].idxmin()
    x = df_eps.loc[idx_eta_min_structure, 'coor_x_sig_sr_1L']
    y = df_eps.loc[idx_eta_min_structure, 'coor_y_sig_sr_1L']
    z = df_eps.loc[idx_eta_min_structure, 'coor_z_sig_sr_1L']

    #TODO: At the moment I extract the coordinates of the first reinfoecement layer, x,y coordinate is exactly the same, sonly the z coordinate is a bit off then
    # TODO: find the right z coordinate, then this also includes if it is in top or bottom layer...

    # Count_ mu values smaller than 1
    GP_count= df_eps[df_eps['eta_min_GP'] < 1]['eta_min_GP'].count()


    # write a dict with results and print
    res_concrete={'eta_min_c' : [eta_min_structure], 'x_c' : [x], 'y_c' : [y], 'z_c' : [z], 'Location_c':[location], 'GP_count_c': [GP_count]}

    if return_type=='dict':
        res=res_concrete
    elif return_type=='df':
        res= pd.DataFrame(res_concrete)
    else:
        raise Exception("Only return types dict and df are implemented. Define one of these types.")
    
    return res



def steel_bending_verification(structure=None, results = None, step=None, return_type='dict', verbalise=True):

    '''
    This function evaluates the structure results with respect to reinforcement bending. 
    It caluclates eta based on the ration between the fsu and the stress in the individual reinforcement (sig_sr).

    Parameters:
    structure: structure object
        a structure object defined after see function ....
    results: dict
        Contains the analysis results for each evaluation step.
    step: string
        the name of the analysis step (e.g. "step_4")
    return type: string
        the type of the return value. Possible return types are dict and df. The default is dict.
    verbalise: bool
        Flag that defines weather the function should print progress information (if set to True) 
        or should run without printing any progress information (if set to False).

    
    Returns:
    res: dict or pandas dataframe
        The function returns a dict with the verification results 
        The dict contains the following keys: 'eta_min_s', 'x_s', 'y_s', 'z_s', 'Location_s', 'GP_count_s'
    '''
    
    # if not results but the structure is given we first extract the results dict from the structure pkl file
    if results == None:
        if structure==None:
            raise Exception("You have not provided a structure of results. You have to provide at one!")

        results = structure.results
        if verbalise:
            print('The structure was converted to results dict')

    # extract and calculate steel bending results for a structure
    df_steel = pd.DataFrame({'sig_sr_1L':list(results[step]['GP']['sig_sr_1L'].values()),
                        'sig_sr_2L':list(results[step]['GP']['sig_sr_2L'].values()),
                        'sig_sr_3L':list(results[step]['GP']['sig_sr_3L'].values()),
                        'sig_sr_4L':list(results[step]['GP']['sig_sr_4L'].values()),
                        'coor_x_sig_sr_1L':list(results[step]['GP']['coor_x_sig_sr_1L'].values()),
                        'coor_y_sig_sr_1L':list(results[step]['GP']['coor_y_sig_sr_1L'].values()),
                        'coor_z_sig_sr_1L':list(results[step]['GP']['coor_z_sig_sr_1L'].values()),
               } )


    #define calculation of eta
    def calculate_eta_ifPos(x):
        if x > 0:
            return 720/x #fsu/sig_sr #720 MPa; N/mm2
        #TODO: Get fsu from somewhere (when constant, or when different for different elemens -> Marius will provide a fsu value for each sigma value (for each layer one))
        #TODO: Check for factors that we might need to apply 1.35 to rrgese fsy fsu values??..
        else:
            return None
    
    # Calculate eta for for each GP an top and bot, for directions 1 and 2 (--> 4 times per GP) 
    df_steel['eta_1']=df_steel['sig_sr_1L'].apply(calculate_eta_ifPos)
    df_steel['eta_2']=df_steel['sig_sr_2L'].apply(calculate_eta_ifPos)
    df_steel['eta_3']=df_steel['sig_sr_3L'].apply(calculate_eta_ifPos)
    df_steel['eta_4']=df_steel['sig_sr_4L'].apply(calculate_eta_ifPos)

    
    if verbalise:
        #print number of elements
        print('Number of elements: ',len(df_steel['eta_1'])/4)

    # get minimum eta fpr each Gaus Point
    df_steel['eta_min_GP'] = df_steel[['eta_1',  'eta_2',  'eta_3', 'eta_4']].min(axis=1)
    # Get Minimum eta value of structure
    eta_min_structure = df_steel['eta_min_GP'].min()

    # Find the location (columnname) of the minimum eta value
    location = df_steel[['eta_1',  'eta_2',  'eta_3', 'eta_4']][df_steel == eta_min_structure].stack().index.tolist()[0][1]

    #TODO: include that it is tracked which is the minimum layer 1,2,3 or 4 (to be also predicted)?
    # TODO: one could include all coordinates of all layers and save the right z value than the z bvalue is more accurate!


    # Getting coordinates of min position
    idx_eta_min_structure = df_steel['eta_min_GP'].idxmin()
    x = df_steel.loc[idx_eta_min_structure, 'coor_x_sig_sr_1L']
    y = df_steel.loc[idx_eta_min_structure, 'coor_y_sig_sr_1L']
    z = df_steel.loc[idx_eta_min_structure, 'coor_z_sig_sr_1L']

    # Count_ eta values smaller than 1
    GP_count= df_steel[df_steel['eta_min_GP'] < 1]['eta_min_GP'].count()


    # write a dict with results and print
    res_steel={'eta_min_s' : [eta_min_structure], 'x_s' : [x], 'y_s' : [y], 'z_s' : [z], 'Location_s':[location], 'GP_count_s': [GP_count]}
    if return_type=='dict':
        res=res_steel
    elif return_type=='df':
        res= pd.DataFrame(res_steel)
    else:
        raise Exception("Only return types dict and df are implemented. Define one of these types.")

    return res



def steel_bending_verification_old(structure=None, results = None, step=None, return_type='dict', verbalise=True):

    '''
    This function evaluates the structure results with respect to reinforcement bending. 
    It caluclates eta based on the ration between the fsu and the stress in the individual reinforcement (sig_sr).

    Parameters:
    structure: structure object
        a structure object defined after see function ....
    results: dict
        Contains the analysis results for each evaluation step.
    step: string
        the name of the analysis step (e.g. "step_4")
    return type: string
        the type of the return value. Possible return types are dict and df. The default is dict.
    verbalise: bool
        Flag that defines weather the function should print progress information (if set to True) 
        or should run without printing any progress information (if set to False).

    
    Returns:
    res: dict or pandas dataframe
        The function returns a dict with the verification results 
        The dict contains the following keys: 'eta_min_s', 'x_s', 'y_s', 'z_s', 'Location_s', 'GP_count_s'
    '''
    
    # if not results but the structure is given we first extract the results dict from the structure pkl file
    if results == None:
        if structure==None:
            raise Exception("You have not provided a structure of results. You have to provide at one!")

        results = structure.results
        if verbalise:
            print('The structure was converted to results dict')

    # extract and calculate steel bending results for a structure
    df_steel = pd.DataFrame({'sig_sr_1L_x':list(results[step]['GP']['sig_sr_1L_x'].values()),
                        'sig_sr_1L_y':list(results[step]['GP']['sig_sr_1L_y'].values()),
                        'sig_sr_2L_x':list(results[step]['GP']['sig_sr_2L_x'].values()),
                        'sig_sr_2L_y':list(results[step]['GP']['sig_sr_2L_y'].values()),
                        'sig_sr_3L_x':list(results[step]['GP']['sig_sr_3L_x'].values()),
                        'sig_sr_3L_y':list(results[step]['GP']['sig_sr_3L_y'].values()),
                        'sig_sr_4L_x':list(results[step]['GP']['sig_sr_4L_x'].values()),
                        'sig_sr_4L_y':list(results[step]['GP']['sig_sr_4L_y'].values()),
                        'coor_x_sig_sr_1L':list(results[step]['GP']['coor_x_sig_sr_1L'].values()),
                        'coor_y_sig_sr_1L':list(results[step]['GP']['coor_y_sig_sr_1L'].values()),
                        'coor_z_sig_sr_1L':list(results[step]['GP']['coor_z_sig_sr_1L'].values()),
               } )


    #define calculation of eta
    def calculate_eta_ifPos(x):
        if x > 0:
            return 720/x #fsu/sig_sr #720 MPa; N/mm2
        #TODO: Get fsu from somewhere (when constant, or when different for different elemens -> Marius will provide a fsu value for each sigma value (for each layer one))
        #TODO: Check for factors that we might need to apply 1.35 to rrgese fsy fsu values??..
        else:
            return None
    
    # Calculate eta for for each GP an top and bot, for directions 1 and 2 (--> 4 times per GP) 
    df_steel['eta_1_x']=df_steel['sig_sr_1L_x'].apply(calculate_eta_ifPos)
    df_steel['eta_1_y']=df_steel['sig_sr_1L_y'].apply(calculate_eta_ifPos)
    df_steel['eta_2_x']=df_steel['sig_sr_2L_x'].apply(calculate_eta_ifPos)
    df_steel['eta_2_y']=df_steel['sig_sr_2L_y'].apply(calculate_eta_ifPos)
    df_steel['eta_3_x']=df_steel['sig_sr_3L_x'].apply(calculate_eta_ifPos)
    df_steel['eta_3_y']=df_steel['sig_sr_3L_y'].apply(calculate_eta_ifPos)
    df_steel['eta_4_x']=df_steel['sig_sr_4L_x'].apply(calculate_eta_ifPos)
    df_steel['eta_4_y']=df_steel['sig_sr_4L_y'].apply(calculate_eta_ifPos)
    
    if verbalise:
        #print number of elements
        print('Number of elements: ',len(df_steel['eta_1_x'])/4)

    # get minimum eta fpr each Gaus Point
    df_steel['eta_min_GP'] = df_steel[['eta_1_x', 'eta_1_y', 'eta_2_x', 'eta_2_y', 'eta_3_x', 'eta_3_y', 'eta_4_x', 'eta_4_y']].min(axis=1)
    # Get Minimum eta value of structure
    eta_min_structure = df_steel['eta_min_GP'].min()

    # Find the location (columnname) of the minimum eta value
    location = df_steel[['eta_1_x', 'eta_1_y', 'eta_2_x', 'eta_2_y', 'eta_3_x', 'eta_3_y', 'eta_4_x', 'eta_4_y']][df_steel == eta_min_structure].stack().index.tolist()[0][1]

    #TODO: include that it is tracked which is the minimum layer 1,2,3 or 4 (to be also predicted)?
    # TODO: one could include all coordinates of all layers and save the right z value than the z bvalue is more accurate!


    # Getting coordinates of min position
    idx_eta_min_structure = df_steel['eta_min_GP'].idxmin()
    x = df_steel.loc[idx_eta_min_structure, 'coor_x_sig_sr_1L']
    y = df_steel.loc[idx_eta_min_structure, 'coor_y_sig_sr_1L']
    z = df_steel.loc[idx_eta_min_structure, 'coor_z_sig_sr_1L']

    # Count_ eta values smaller than 1
    GP_count= df_steel[df_steel['eta_min_GP'] < 1]['eta_min_GP'].count()


    # write a dict with results and print
    res_steel={'eta_min_s' : [eta_min_structure], 'x_s' : [x], 'y_s' : [y], 'z_s' : [z], 'Location_s':[location], 'GP_count_s': [GP_count]}
    if return_type=='dict':
        res=res_steel
    elif return_type=='df':
        res= pd.DataFrame(res_steel)
    else:
        raise Exception("Only return types dict and df are implemented. Define one of these types.")

    return res

def concrete_bending_verification_stresses(structure=None, results = None, step=None, return_type='dict', verbalise=True):

    '''
    This function evaluates the structure results with respect to concrete bending. 
    It caluclates eta_stresses based on the ratio between the calculated fcd_eff and the caluclated acting stresses (sigma). 
    fcd_eff is the effective concrete compressive stress, which is reduced due to the corresponnding strainstate (eps_1) at the individual position.

    Parameters:
    structure: structure object
        a structure object defined after see function ....
    results: dict
        Contains the analysis results for each evaluation step.
    step: string
        the name of the analysis step (e.g. "step_4")
    return type: string
        the type of the return value. Possible return types are dict and df. The default is dict.
    verbalise: bool
        Flag that defines weather the function should print progress information (if set to True) 
        or should run without printing any progress information (if set to False).

    
    Returns:
    res: dict or pandas dataframe
        The function returns a dict with the verification results 
        The dict contains the following keys: 'eta_min_c_stresses', 'x_c_stresses', 'y_c_stresses', 'z_c_stresses', 'Location_c_stresses', 'GP_count_c'
    '''

    # if not results but the structure is given we first extract the results dict from the structure pkl file
    if results == None:
        if structure==None:
            raise Exception("You have not provided a structure of results. You have to provide at one!")
        
        results = structure.results
        if verbalise:
            print('The structure was converted to results dict')


    # save relevant values from dict into a data frame for easyer handling
    df_sig = pd.DataFrame({'sig_x_top':list(results[step]['GP']['sig_x_top'].values()),
                         'sig_y_top':list(results[step]['GP']['sig_y_top'].values()),
                         'sig_x_bot':list(results[step]['GP']['sig_x_bot'].values()),
                         'sig_y_bot':list(results[step]['GP']['sig_y_bot'].values()),

                         'coor_x_sig_sr_1L':list(results[step]['GP']['coor_x_sig_sr_1L'].values()),
                        'coor_y_sig_sr_1L':list(results[step]['GP']['coor_y_sig_sr_1L'].values()),
                         'coor_z_sig_sr_1L':list(results[step]['GP']['coor_z_sig_sr_1L'].values()),


                        'fcc_eff_top':list(results[step]['GP']['fcc_eff_top'].values()),
                        'fcc_eff_bot':list(results[step]['GP']['fcc_eff_bot'].values()),
                         } )       


    #define calculation of eta_stresses
    def calculate_eta_sig_ifNeg(row):
        # Extracting the values for fcc_eff_top/bot and sig_x_top/bot from the row
        x = row.iloc[0]
        y = row.iloc[1]
        # Performing the division if sig is less than 0
        if y < 0:
            return -x / y  # fcc_eff/sig
        else:
            return None  # Returning None if the condition is not met

    # Calculate eta_stresses for for each GP an top and bot, for directions x and y (--> 4 times per GP) 
    df_sig['eta_x_top_stresses']=df_sig[['fcc_eff_top','sig_x_top']].apply(calculate_eta_sig_ifNeg,axis=1)
    df_sig['eta_y_top_stresses']=df_sig[['fcc_eff_top','sig_y_top']].apply(calculate_eta_sig_ifNeg,axis=1)
    df_sig['eta_x_bot_stresses']=df_sig[['fcc_eff_bot','sig_x_bot']].apply(calculate_eta_sig_ifNeg,axis=1)
    df_sig['eta_y_bot_stresses']=df_sig[['fcc_eff_bot','sig_y_bot']].apply(calculate_eta_sig_ifNeg,axis=1)


    # get minimum eta for each Gaus Point
    df_sig['eta_min_GP_stresses'] = df_sig[['eta_x_top_stresses', 'eta_y_top_stresses', 'eta_x_bot_stresses', 'eta_y_bot_stresses']].min(axis=1)
    # Get Minimum eta value of structure
    eta_min_structure_stresses = df_sig['eta_min_GP_stresses'].min()
    #min_value = df_eps[['eta_1_bot', 'eta_3_bot', 'eta_1_top', 'eta_3_top']].min().min()

    # Find the location (columnname) of the minimum eta value
    location = df_sig[['eta_x_top_stresses', 'eta_y_top_stresses', 'eta_x_bot_stresses', 'eta_y_bot_stresses']][df_sig == eta_min_structure_stresses].stack().index.tolist()[0][1]
    

    # Getting coordinates of min position
    idx_eta_min_structure_stresses = df_sig['eta_min_GP_stresses'].idxmin()
    x = df_sig.loc[idx_eta_min_structure_stresses, 'coor_x_sig_sr_1L']
    y = df_sig.loc[idx_eta_min_structure_stresses, 'coor_y_sig_sr_1L']
    z = df_sig.loc[idx_eta_min_structure_stresses, 'coor_z_sig_sr_1L']

    #TODO: At the moment I extract the coordinates of the first reinfoecement layer, x,y coordinate is exactly the same, sonly the z coordinate is a bit off then
    # TODO: find the right z coordinate, then this also includes if it is in top or bottom layer...

    # Count_ mu values smaller than 1
    # GP_count= df_eps[df_eps['eta_min_GP'] < 1]['eta_min_GP'].count()


    # write a dict with results and print
    res_concrete={'eta_min_c_stresses' : [eta_min_structure_stresses], 'x_c_stresses' : [x], 'y_c_stresses' : [y], 'z_c_stresses' : [z], 'Location_c_stresses':[location]} #, 'GP_count_c': [GP_count]}

    if return_type=='dict':
        res=res_concrete
    elif return_type=='df':
        res= pd.DataFrame(res_concrete)
    else:
        raise Exception("Only return types dict and df are implemented. Define one of these types.")
    
    return res

def concrete_shear_verification(structure = None, results=None, step=None, return_type='dict', verbalise=True ):
    '''
    Performs the shear verification for the structure.
    Calculating the minimum eta_v of the structure and evaluating the critical position.

    Parameters:
    structure: structure object
        a structure object defined after see function ....
    results: dict
        Contains the analysis results for each evaluation step.
    step: string
        the name of the analysis step (e.g. "step_4")
    return type: string
        the type of the return value. Possible return types are "dict" and "df". The default is dict.
    verbalise: bool
        Flag that defines weather the function should print progress information (if set to True) 
        or should run without printing any progress information (if set to False).

    
    Returns:
    res: dict or pandas dataframe
        The function returns a dict with the shear verification results 
        The dict contains the following keys: 'eta_min_shear', 'x_c_shear', 'y_c_shear', 'z_c_shear', 'Location_c_shear', 'GP_count_shear'
    '''

    # if not results but the structure is given we first extract the results dict from the structure pkl file
    if results == None:
        if structure == None:
            raise Exception("You have not provided a structure of results. You have to provide at one!")
        
        results = structure.results
        if verbalise:
            print('The structure was converted to results dict')
    
    # get shear verification already perfromed on element level
    eta_v_dict=results[step]['element']['eta_v']
    eps_06d_loc_decisive_dict=results[step]['element']['eps_06d_loc_decisive']
    centroid_dict=results[step]['element']['centroid']

    # Find the element with the minimum eta_v value (disregard Nones)
    eta_v_dict_filtered = {key: value for key, value in eta_v_dict.items() if value is not None} # Filter out None values
    min_element = min(eta_v_dict_filtered, key=eta_v_dict_filtered.get)
    eta_min_shear = eta_v_dict_filtered[min_element]
    element_count = sum(1 for value in eta_v_dict_filtered.values() if value < 1) # Count values less than 1 in the filtered dictionary
    eps_06d_loc_decisive=eps_06d_loc_decisive_dict[min_element]
    centroid=centroid_dict[min_element]
    x=centroid[0]
    y=centroid[1]
    z=centroid[2]

    # write a dict with results and print
    res_concrete={'eta_min_shear' : [eta_min_shear], 'x_c_shear' : [x], 'y_c_shear' : [y], 'z_c_shear' : [z], 'Location_c_shear':[eps_06d_loc_decisive], 'element_count_shear': [element_count]}

    if return_type=='dict':
        res=res_concrete
    elif return_type=='df':
        res= pd.DataFrame(res_concrete)
    else:
        raise Exception("Only return types dict and df are implemented. Define one of these types.")
    
    return res
    




def calc_eta(idx_s, start_id, end_id, step, extract_from ='results', folder_name='CFBData', with_eta_stresses=False, verbalise=True):

    '''
    This function iterates from start_id to end_id. Opens the corresponding file. 
    And calls the result extraction and verification functions for the bending verification.
    It caluclates eta based on the ration between the fsu and the stress in the individual reinforcement (sig_sr).

    Parameters:
    idx_s: integer
        Index of sampling. File identifier.
    start_id: integer
        Structure ID where the function should start iterating.
    end_id: integer
        Structure ID where the function should end iterating.
    step: string
        the name of the analysis step (e.g. "step_4")  
    extract_from: string
        From what type of file should the analysis results be extracted. Possible is "structure" and "results" file.
    folder_name: string
        The name of the folder where the results are located in
    with_eta_stresses: bool
        Flag that defines weather the eta_stresses should also be caluclated for concrete in bending
    verbalise: bool
        Flag that defines weather the function should print progress information (if set to True) 
        or should run without printing any progress information (if set to False).

    
    Returns:
    df_res: pandas dataframe
        The function returns a dataframe with the verification results combined of all structures from start_id to end_id.
        Each row of the dataframe coresponds to one structure.
    '''

    # Initialize a dictionary to store the extracted data (dict of empty lists)
    df_res=None
    noRes_count=0
    error_count=0
    error_ids=[]
    

    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)

    for ID in range(start_id,end_id+1):
        if verbalise:
            print('ID: ',ID)

        #init no file variable
        nofile=False
        # error_flag=False

        #get structure or results file
        if extract_from=='structure':
            if verbalise:
                print("Extracting from structure.pkl file")
            # construct file path
            filepath=folder_path+'\\{}_Batch\\{}_{}_CFB\\{}_{}_structure.pkl'.format(idx_s,idx_s,ID,idx_s,ID)
            #if a structure pickle file exists
            if os.path.exists(filepath):
                with open(filepath, "rb") as pickle_file:
                    srtructure = pickle.load(pickle_file)
                results=srtructure.results
            else: 
                if verbalise:
                    print('The structure.pkl file does not exist: ', filepath)
                nofile=True

        elif extract_from=='results':
            if verbalise:
                print("Extracting from results.json file")
            #TODO: use the get_results(9 function for this) but then the path not fund handling has to be checked
            filepath=folder_path+'\\{}_Batch\\{}_{}_CFB\\{}_{}_analysisResults.json'.format(idx_s,idx_s,ID,idx_s,ID)
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    results = json.load(file)
            else:
                if verbalise:
                    print('The rsults.json file does not exist: ', filepath)
                nofile=True
        else:
            raise Exception("The string given to the extract_from input is invalid. Valid strings are results and structure.")
        
        #if results/ structure file exists
        if not nofile:
            if len(results[step]) >0: 
                if results[step]=='ERROR':
                    error_ids.append(ID)
                    error_count+=1

                    if verbalise:
                        print('The rsults.json file exits. However an error happened during the NLFE Analysis.')
                        print('eta_min_c and eta_min_s are set to 0!')

                    res_steel=pd.DataFrame({'eta_min_s' : [0.], 'x_s' : [None], 'y_s' : [None], 'z_s' : [None], 'Location_s':None, 'GP_count_s': [None]})
                    res_concrete=pd.DataFrame({'eta_min_c' : [0.], 'x_c' : [None], 'y_c' : [None], 'z_c' : [None], 'Location_c':[None], 'GP_count_c': [None]})
                    res_conc_shear=pd.DataFrame({'eta_min_shear' : [0.]})

                    if with_eta_stresses:
                        res_concrete_stresses=pd.DataFrame({'eta_min_c_stresses' : [0.], 'x_c_stresses' : [None], 'y_c_stresses' : [None], 'z_c_stresses' : [None], 'Location_c_stresses':[None]})

                    if with_eta_stresses:
                        df_c_s=pd.concat([res_steel,res_concrete, res_conc_shear, res_concrete_stresses], axis=1)
                    else:
                        df_c_s=pd.concat([res_steel,res_concrete, res_conc_shear], axis=1)

                    df_c_s['ID']=ID

                else:
                    # extract max values from the analysis results of this structre 
                    df_steel=steel_bending_verification(results=results, step=step, return_type='df',verbalise=verbalise)
                    df_conc=concrete_bending_verification(results=results, step=step, return_type='df', verbalise=verbalise)
                    df_conc_shear=concrete_shear_verification(results=results, step=step, return_type='df', verbalise=verbalise)

                    if with_eta_stresses:
                        df_conc_stresses=concrete_bending_verification_stresses(results=results, step=step, return_type='df',verbalise=verbalise)
                    
                    if with_eta_stresses:
                        df_c_s=pd.concat([df_steel,df_conc, df_conc_shear, df_conc_stresses], axis=1)
                    else:
                        df_c_s=pd.concat([df_steel,df_conc, df_conc_shear], axis=1)

                    df_c_s['ID']=ID
            else:
                raise Exception("The results/ structure file exists however the defined step is empty. No analysis results can be extracted.")

            if ID==start_id:
                df_res=df_c_s
            else:
                df_res=pd.concat([df_res,df_c_s])


        # if no structure pickle or result json file exists (due to an error (not divergence) or simply wrong path)
        if nofile:
            print('The results could not extracted, as no results file was available.')
            raise Exception("The results/structure file does not exist. Check the path or rerun the analysis. Path: "+filepath)
            # TODO: create an empty row, so I can still analyse the other ones even though one was skipped?
    
    print('There were {} Structured that resulted in an error during analysis. (Error IDs: {})'.format(error_count,error_ids))
    return df_res


def get_results(idx_s, ID, step=None, extract_from ='results', folder_name='CFBData', verbalise=True):
    '''
    This function loads the results dict for bridge with the ID from the sampling batch idx_s. If a step is provided only the results dict for that step is returned.

    Parameters:
    idx_s: integer
        Index of sampling. File identifier.
    ID: integer
        Structure ID.
    step: string
        the name of the analysis step (e.g. "step_4")  
    extract_from: string
        From what type of file should the analysis results be extracted. Possible is from a "results" file.
    folder_name: string
        The name of the folder where the results are located in
    verbalise: bool
        Flag that defines weather the function should print progress information (if set to True) 
        or should run without printing any progress information (if set to False).

    
    Returns:
    results: dict
        The function returns the results dict for the requested bridge identifiers (and load step).
    '''
        
    #construct path
    current_directory = os.getcwd()
    filepath=current_directory+'\\CFBData\\{}_Batch\\{}_{}_CFB\\{}_{}_analysisResults.json'.format(idx_s,idx_s,ID,idx_s,ID)
    if verbalise:
        print('filepath: ',filepath)

    # Load results dict form json file
    if os.path.exists(filepath):
        print('Path found.')
        with open(filepath, 'r') as file:
            results = json.load(file)
            if verbalise:
                print('Results imported')
            if not step==None:
                results=results[step]
             
    else: 
        if verbalise:
            print('Path not found.')
        results=None


    return results

def get_structure(idx_s, ID, folder_name='CFBData', extract_from='pickle', verbalise=True):
    '''
    This function loads the structure object with the ID from the sampling batch idx_s. If a step is provided only the results dict for that step is returned.

    Parameters:
    idx_s: integer
        Index of sampling. File identifier.
    ID: integer
        Structure ID.
    folder_name: string
        The name of the folder where the results are located in
    verbalise: bool
        Flag that defines weather the function should print progress information (if set to True) 
        or should run without printing any progress information (if set to False).

    
    Returns:
    structure: obj
        returns the impotred structure object.
    '''
        
    #construct path
    current_directory = os.getcwd()
    filepath=current_directory+'\\CFBData\\{}_Batch\\{}_{}_CFB\\{}_{}_structure.pkl'.format(idx_s,idx_s,ID,idx_s,ID)
    if verbalise:
        print('filepath: ',filepath)

    # Load results dict form json file
    if os.path.exists(filepath):
        print('Path found.')
        with open(filepath, 'rb') as file:
            structure = pickle.load(file)
            if verbalise:
                print('Structure imported')

             
    else: 
        if verbalise:
            print('Path not found.')
        structure=None


    return structure



