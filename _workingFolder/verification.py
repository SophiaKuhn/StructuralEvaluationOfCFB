 # Author(s): Sophia Kuhn (ETH Zurich)


import pandas as pd
import os
import json 

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



def calc_eta(idx_s, start_id, end_id, step, extract_from ='results', folder_name='CFBData', verbalise=True):

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

                    res_concrete=pd.DataFrame({'eta_min_c' : [0.], 'x_c' : [None], 'y_c' : [None], 'z_c' : [None], 'Location_c':[None], 'GP_count_c': [None]})
                    res_steel=pd.DataFrame({'eta_min_s' : [0.], 'x_s' : [None], 'y_s' : [None], 'z_s' : [None], 'Location_s':None, 'GP_count_s': [None]})
                    df_c_s=pd.concat([res_steel,res_concrete], axis=1)
                    df_c_s['ID']=ID

                else:
                    # extract max values from the analysis results of this structre 
                    df_conc=concrete_bending_verification(results=results, step=step, return_type='df', verbalise=verbalise)
                    df_steel=steel_bending_verification(results=results, step=step, return_type='df',verbalise=verbalise)
                    df_c_s=pd.concat([df_steel,df_conc], axis=1)
                    df_c_s['ID']=ID
            else:
                raise Exception("The results/ structure file exists however the defined step is empty. No analysis results can be extracted.")

            if ID==start_id:
                df_res=df_c_s
            else:
                df_res=pd.concat([df_res,df_c_s])

        # # if no structure pickle or result json file exists (due to calculation error or simply wrong path)
        # if error_flag:
        #     error_count+=1

        #     if verbalise:
        #         print('eta_min_c and eta_min_s are set to 0!')
        #     res_concrete=pd.DataFrame({'eta_min_c' : [0.], 'x_c' : [None], 'y_c' : [None], 'z_c' : [None], 'Location_c':[None], 'GP_count_c': [None]})
        #     res_steel=pd.DataFrame({'eta_min_s' : [0.], 'x_s' : [None], 'y_s' : [None], 'z_s' : [None], 'Location_s':None, 'GP_count_s': [None]})
        #     df_c_s=pd.concat([res_steel,res_concrete], axis=1)
        #     df_c_s['ID']=ID
            
        #     if ID==start_id:
        #         df_res=df_c_s
        #     else:
        #         df_res=pd.concat([df_res,df_c_s])

        # if no structure pickle or result json file exists (due to an error (not divergence) or simply wrong path)
        if nofile:
            print('The results could not extracted, as no results file was available.')
            raise Exception("The results/structure file does not exist. Check the path or rerun the analysis. Path: "+filepath)
            # TODO: create an empty row, so I can still analyse the other ones even though one was skipped?
    
    print('There were {} Structured that resulted in an error during analysis. (Error IDs: {})'.format(error_count,error_ids))
    return df_res