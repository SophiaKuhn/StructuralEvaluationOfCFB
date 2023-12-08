


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# NLFE Script
# Author: Marius Weber (ETHZ, HSLU T&A) and Sophia Kuhn (ETHZ)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# Import packages
# ------------------------------------------------------------------------------
#------------------------------------------------------------------------------
from compas_fea.cad import rhino
from compas_fea.structure import ElasticIsotropic
from compas_fea.structure import MPCStiff
from compas_fea.structure import CMMUsermat
from compas_fea.structure import ElementProperties as Properties
from compas_fea.structure import GeneralStep
from compas_fea.structure import GravityLoad
from compas_fea.structure import AreaLoad
from compas_fea.structure import PointLoad
from compas_fea.structure import GeneralDisplacement
from compas_fea.structure import FixedDisplacement
from compas_fea.structure import FixedDisplacementXX
from compas_fea.structure import FixedDisplacementYY
from compas_fea.structure import FixedDisplacementZZ
from compas_fea.structure import PinnedDisplacement
from compas_fea.structure import RollerDisplacementX
#from compas_fea.structure import RollerDispslacementY
from compas_fea.structure import RollerDisplacementZ
from compas_fea.structure import RollerDisplacementXY
from compas_fea.structure import RollerDisplacementYZ
from compas_fea.structure import RollerDisplacementXZ
from compas_fea.structure import ShellSection
from compas_fea.structure import MPCSection
from compas_fea.structure import Structure
from strucenglib.prepost_functions import calc_loc_coor
from strucenglib.prepost_functions import plot_loc_axes
from strucenglib.prepost_functions import plot_nr_elem
from strucenglib.prepost_functions import plot_nr_nodes
from strucenglib.prepost_functions import area_load_generator_elements
from strucenglib.prepost_functions import Normalspurbahnverkehr_load_generator
from strucenglib.prepost_functions import verification
from strucenglib.sandwichmodel import sandwichmodel_main as SMM

#New Functions
from export import read_csv_to_dict,extract_numbers_from_string #utility functions
from export import delete_all, joinMeshes_inShellLayers #rhino functions
from export import save_to_pickle,save_to_json #export_results #export functions

#from strucenglib.prepost_functions.earthPressure_load_generator import earthPressure_calculator,earthPressure_load_generator
from earthPressure_load_generator import earthPressure_backfill_generator, earthPressure_liveload_generator #,earthPressure_load_generator

import rhinoscriptsyntax as rs
import time
import Rhino as r
import scriptcontext as sc
import csv
import os
import math as m



#--------- read Parameter from sampled csv file-------------------------

# define folder where samples and geo is saved
#folder='C:\\CFBData'

# define sampling iteration (= Batch number)
idx_s = 1
## Read current sampling index (idx_s)
#file_path=folder+'\\00_Sampled\\\current_idx_s.csv'
#idx_s = read_csv_to_dict(file_path)['idx_s'][0]
#print('Sampling Iteration Index: idx_s = ',idx_s)


# Read corresponding csv files, where all samples are saved
# read all sampled variables to a dict
current_directory = os.getcwd()
folder_name='CFBData'
folder_path = os.path.join(current_directory, folder_name)
csv_file_path = folder_path+ '\\{}_Batch\\{}_CFBSamples.csv'.format(idx_s,idx_s)
data_dict = read_csv_to_dict(csv_file_path)


#determine number of samples of the current sampling iteration
n_samples=len(data_dict[""])-1
print('n_samples: ',n_samples)



#---------------------------------------------------------------------------------------
#--------------iterate through the generated samples------------------------------------
#---------------------------------------------------------------------------------------
start = 0
end = 0#n_samples
for i in range(start,end+1):
    
            
    #------------Deleate everything in rhino file-----------------
    # deleate everything for next iteration
    # exept the exept_layer (but still all objects in the layer are deleted)
    delete_all(exept_layer="Default")
    
    #------------Import Geometry file-----------------------------------
    ID=int(data_dict[""][i])
    # import geometry, layers and UserText from generated 3dm-files
    filepath=folder_path+'\\{}_Batch\\{}_{}_CFB\\geo.3dm'.format(idx_s,idx_s,ID) #{}_{}_geo.3dm'.format(idx_s,idx_s,ID)
    rs.Command("!_-Import \"" + filepath + "\" -Enter -Enter")
    
    
    #------------check if IDs match-----------------
    # here I check if the ID of the sampled row matches with the geometry file
    # get both ID values
    ID_check=int(float(rs.GetDocumentUserText('id')))
    
    # Check if their values are equal
    if ID == ID_check:
        print("Both 'ID' and 'ID_check' have the same value.")
    else:
        raise ValueError("Error: 'ID' and 'ID_check' have different values.")
        
    print('ID: ',ID)
    
    #------------extract parameters from sampled dict or UserText-----------------
    # Import/get parameter for current iteration
    # --01_global parameter (from sampler csv --> imported as dict)--
    #geometric parameter
    h_w = float(data_dict["h_w"][i])
    b1 = float(data_dict["b1"][i])
    t_p = float(data_dict["t_p"][i])
    t_w = float(data_dict["t_w"][i])
    tmin=min(t_p,t_w)
    
    oo = int(data_dict["oo"][i])
    uu = int(data_dict["uu"][i])
    # Matrial properties: concrete
    fcc = int(float(data_dict["fcc"][i]))
    ecu = float(data_dict["ecu"][i])
    # Matrial properties: reinforcement
    fsy = int(float(data_dict["fsy"][i]))
    fsu = int(float(data_dict["fsu"][i]))
    esu = float(data_dict["esu"][i])
    # load parameter
    s=float(data_dict["s"][i])
    beta=float(data_dict["beta"][i])
    
    h_G=float(data_dict["h_G"][i])
    gamma_E=float(data_dict["gamma_E"][i])
    phi_k=float(data_dict["phi_k"][i])
    
    q_Gl = float(data_dict["q_Gl"][i])
    b_Bs = float(data_dict["b_Bs"][i])
    Q_k = float(data_dict["Q_k"][i])
    

   
   
    # --02_global parameter calculated in GH model-- (these are calculated dependent on sampled parameters) (saved in DocUserText)
    h_w_m=float(rs.GetDocumentUserText('h_w_m'))
    L_el= float(rs.GetDocumentUserText('L_el'))
    
    
    # ---------Initialise---------------------
    name = 'NLFE_CFB_{}_{}'.format(idx_s, ID)
    path = 'C:\Temp\\'
    mdl = Structure(name=name, path=path)

    # Structure
    mdl = Structure(name=name, path=path)
    
    
    #------------add sets from Rhino file --------------------
    
    #get all layerNames
    layer_names=rs.LayerNames()
    
    #--Shell Elements--
    #extract shell elements layer names
    group_str="elset_plate_"
    layers_deck = [l for l in layer_names if group_str in l]
    group_str="elset_wall_1"
    layers_wall1 = [l for l in layer_names if group_str in l]
    group_str="elset_wall_2"
    layers_wall2= [l for l in layer_names if group_str in l]
    
    layers_walls = layers_wall1 + layers_wall2
    layers_shell=layers_deck +  layers_walls
    
    # join meshes (backend does not run for layers with multiple elements in it!)
    joinMeshes_inShellLayers(layers_shell)
    rhino.add_nodes_elements_from_layers(mdl, mesh_type='ShellElement', layers= layers_shell )
    
    
    
    #-- MPC Elements--
    rhino.add_nodes_elements_from_layers(mdl, line_type='MPCElement', layers='elset_mpcs')
    
    
    
    #-- Sets for constraints--
    layers_support= ['nset_support_wall1', 'nset_support_wall2' ]
    rhino.add_sets_from_layers(mdl, layers=layers_support )
    
    

    # -------------add materials, sections, properties --------------------
    # Shell Sections, Properties, additional Properties, set local coordiantes (The local z-axed is adjusted by using "object direction" in Rino)
    
    
    
    #### PLATE ####
    semi_loc_coords=calc_loc_coor(layer=layers_deck[0], PORIG=[0,0,0],PXAXS=[1,0,0]) 
    
    #--read out sampled local parameters from the read in dict--
    # reinforcement geometry
    d1 = float(data_dict["d1_plate"][i]) #[mm]
    d2 = float(data_dict["d2_plate"][i]) 
    d3 = float(data_dict["d3_plate"][i])
    d4 = float(data_dict["d4_plate"][i])
    
    spacing = float(data_dict["s_plate"][i]) #[mm]

    #calculate reinforcement 
    as1 = (d1**2 *m.pi)/(4*spacing) #[mm2/mm] Correct unit for input?
    as2 = (d2**2 *m.pi)/(4*spacing) #Todo alterntaive calculate this directly in the sampling script! and read out
    as3 = (d3**2 *m.pi)/(4*spacing)
    as4 = (d4**2 *m.pi)/(4*spacing)
    
    # -- define corresponding material  --
    #TODO oo and uu einlesen
    geo={'R_Rohr':-1, 'rho':0.0000025, 'oo':oo, 'uu':uu}
    concrete={'beton':2, 'fcc':fcc, 'vc':0, 'ecu':ecu, 'k_E':10000, 'theta_b0':2, 'theta_b1':1, 'k_riss':0, 'Entfestigung':0, 'lambdaTS':0.67, 'srmx':1, 'srmy':1, 'Begrenzung':2, 'KritQ':0, 'winkelD':45, 'k_vr':1, 'fswy':500}
    reinf_1L={'stahl':1,'zm':2,'fsy':fsy,'fsu':fsu,'esu':esu,'esv':0.02,'Es':200000,'ka':-1,'kb':-1,'kc':-1,'as':as1,'dm':d1,'psi':90}
    reinf_2L={'stahl':1,'zm':2,'fsy':fsy,'fsu':fsu,'esu':esu,'esv':0.02,'Es':200000,'ka':-1,'kb':-1,'kc':-1,'as':as2,'dm':d2,'psi':0}
    reinf_3L={'stahl':1,'zm':2,'fsy':fsy,'fsu':fsu,'esu':esu,'esv':0.02,'Es':200000,'ka':-1,'kb':-1,'kc':-1,'as':as3,'dm':d3,'psi':0}
    reinf_4L={'stahl':1,'zm':2,'fsy':fsy,'fsu':fsu,'esu':esu,'esv':0.02,'Es':200000,'ka':-1,'kb':-1,'kc':-1,'as':as4,'dm':d4,'psi':90}
    name_mat='plate_element_mat'
    mdl.add(CMMUsermat(name=name_mat, geo=geo, concrete=concrete, reinf_1L=reinf_1L, reinf_2L=reinf_2L, reinf_3L=reinf_3L, reinf_4L=reinf_4L,))
    
    # -- define section -- 
    # calc. min number of nn (nn >= tmin/dmin)
    dmin=min(d1,d2,d3,d4)
    nn_min=tmin/dmin
    name_sec ='plate_element_sec'
    mdl.add(ShellSection(name=name_sec, t=t_p, semi_loc_coords=semi_loc_coords, nn=nn_min))
    
    
    
    for plate in layers_deck:       
        # -- define elset properties -- 
        name_prop = '{}_element_prop'.format(plate)
        mdl.add(Properties(name=name_prop, material=name_mat, section=name_sec, elset=plate))
        
        
        
    #### Walls ####
    #--read out sampled local parameters from the read in dict--
    # reinforcement geometry                
    d1 = float(data_dict["d1_walls"][i]) #[mm]
    d2 = float(data_dict["d2_walls"][i]) 
    d3 = float(data_dict["d3_walls"][i])
    d4 = float(data_dict["d4_walls"][i])
    
    spacing=float(data_dict["s_walls"][i]) #[mm]

    as1 = (d1**2 *m.pi)/(4*spacing) #[mm2/mm] Correct unit for input?
    as2 = (d2**2 *m.pi)/(4*spacing) #Todo alterntaive calculate this directly in the sampling script! and read out
    as3 = (d3**2 *m.pi)/(4*spacing)
    as4 = (d4**2 *m.pi)/(4*spacing)
    
    
    # -- define corresponding material --
    geo={'R_Rohr':-1, 'rho':0.0000025, 'oo':30, 'uu':30}
    concrete={'beton':2, 'fcc':fcc, 'vc':0, 'ecu':ecu, 'k_E':10000, 'theta_b0':2, 'theta_b1':1, 'k_riss':0, 'Entfestigung':0, 'lambdaTS':0.67, 'srmx':1, 'srmy':1, 'Begrenzung':2, 'KritQ':0, 'winkelD':45, 'k_vr':1, 'fswy':500}
    reinf_1L={'stahl':1,'zm':2,'fsy':fsy,'fsu':fsu,'esu':esu,'esv':0.02,'Es':200000,'ka':-1,'kb':-1,'kc':-1,'as':as1,'dm':d1,'psi':90}
    reinf_2L={'stahl':1,'zm':2,'fsy':fsy,'fsu':fsu,'esu':esu,'esv':0.02,'Es':200000,'ka':-1,'kb':-1,'kc':-1,'as':as2,'dm':d2,'psi':0}
    reinf_3L={'stahl':1,'zm':2,'fsy':fsy,'fsu':fsu,'esu':esu,'esv':0.02,'Es':200000,'ka':-1,'kb':-1,'kc':-1,'as':as3,'dm':d3,'psi':0}
    reinf_4L={'stahl':1,'zm':2,'fsy':fsy,'fsu':fsu,'esu':esu,'esv':0.02,'Es':200000,'ka':-1,'kb':-1,'kc':-1,'as':as4,'dm':d4,'psi':90}
    name_mat='wall_element_mat'
    mdl.add(CMMUsermat(name=name_mat, geo=geo, concrete=concrete, reinf_1L=reinf_1L, reinf_2L=reinf_2L, reinf_3L=reinf_3L, reinf_4L=reinf_4L,))
    
    #calc. min number of nn (nn >= tmin/dmin)
    dmin=min(d1,d2,d3,d4)
    nn_min=tmin/dmin
    
    ## Wall 1 ##
    # -- define section-- 
    semi_loc_coords=calc_loc_coor(layer=layers_wall1[0], PORIG=[0,0,-h_w_m],PXAXS=[1,0,0]) 
    name_sec ='wall1_element_sec'
    mdl.add(ShellSection(name=name_sec, t=t_w, semi_loc_coords=semi_loc_coords, nn=nn_min))
    
    for wall in layers_wall1:
        # -- define elset properties -- 
        name_prop = '{}_element_prop'.format(wall)
        mdl.add(Properties(name=name_prop, material=name_mat, section=name_sec, elset=wall))
        
    
    ## Wall 2 ##
    # -- define section-- 
    semi_loc_coords=calc_loc_coor(layer=layers_wall2[0], PORIG=[0,L_el,-h_w_m],PXAXS=[1,0,0])     
    name_sec ='wall2_element_sec'
    mdl.add(ShellSection(name=name_sec, t=t_w, semi_loc_coords=semi_loc_coords, nn=nn_min))
    
    
    for wall in layers_wall2:
        #--define section and property--
        name_prop = '{}_element_prop'.format(wall)
        mdl.add(Properties(name=name_prop, material=name_mat, section=name_sec, elset=wall))





    # MPC Materials
    mdl.add(MPCStiff(name='elset_mpc_element_mat'))
    # MPC Sections
    mdl.add(MPCSection(name='sec_mpc'))
    mdl.add(Properties(name='elset_mpc_element_prop', material='elset_mpc_element_mat', section='sec_mpc', elset='elset_mpcs'))
    
    
    
##   Grafical plots 
#    plot_loc_axes(mdl, axes_scale=50) # Plot Local coordinates 
#    plot_nr_elem(mdl) # Plot Element Numbers
#    plot_nr_nodes(mdl)  # Plot Node Numbers
   
    
    #-------------- Constrains (Displacements)---------------
    # here Marius explicitly wanted the seperate definition of the two pinned sets (keep this)
    mdl.add([GeneralDisplacement(name='nset_pinned_set_disp_1',  x=0, y=0, z=0, xx=0, yy=0, zz=0, nodes='nset_support_wall1'),])
    mdl.add([GeneralDisplacement(name='nset_pinned_set_disp_2',  x=0, y=0, z=0, xx=0, yy=0, zz=0, nodes='nset_support_wall2'),])
    
   
    
    
    # Loads and steps
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    
    #### Gravity Loads
    mdl.add(GravityLoad(name='load_gravity',  x=0.0,  y=0.0,  z=1.0, elements=layers_shell ))
    
    
    
    #### Superimposed Dead Load
    # Load of Gravel layer
    
    #Staendigen lasten
    
    
    # Earth pressure Load generator (characteristic) on Wall 1 and 2
    earth_pressure_backfill_load = earthPressure_backfill_generator(mdl, layers_walls, h_w, t_p, h_G, gamma_E, phi_k ) #[N/mm2]

    #### Live Loads
    # TODO: iterate over ya! to calculate multiple load positions
    #Normalspurverkehr Load generator 
    # h_Pl here die normale platte hoehe/ thickness, ohne voute - read from sampled parameter
    live_load_names=Normalspurbahnverkehr_load_generator(mdl,name='Gleis1', l_Pl=L_el, h_Pl=t_p, s=s*b1, beta=beta, q_Gl=q_Gl, b_Bs=b_Bs, h_Strich=h_G, Q_k=Q_k, y_A=2000)
    
    # Earth pressure load generator (resulting from live load) on wall 1 (only one sided)
    earth_pressure_liveload = earthPressure_liveload_generator(mdl, s*b1, h_w, t_p, phi_k)

    
#    # Steps
#    mdl.add([
#    GeneralStep(name='step_1',   displacements=[ 'nset_pinned_set_disp_1', 'nset_pinned_set_disp_2'] ,  nlgeom=False),
#    GeneralStep(name='step_2',  loads=['load_gravity'] ,   nlgeom=False, increments=1),
#    GeneralStep(name='step_3',  loads=earth_pressure_backfill_load ,   nlgeom=False, increments=1),
#    GeneralStep(name='step_4',  loads=earth_pressure_liveload ,   nlgeom=False, increments=1),
#    GeneralStep(name='step_5',  loads=live_load_names ,   nlgeom=False),
#    ])
#    mdl.steps_order = [ 'step_1', 'step_2', 'step_3', 'step_4', 'step_5']

    # Steps
    loads_list=['load_gravity']
    loads_list= loads_list+earth_pressure_backfill_load +live_load_names +earth_pressure_liveload
    print(loads_list)
    
    
    mdl.add([
    GeneralStep(name='step_1',   displacements=[ 'nset_pinned_set_disp_1', 'nset_pinned_set_disp_2'] ,  nlgeom=False),
    GeneralStep(name='step_2',  loads=loads_list ,   nlgeom=False, increments=1)
    ])
    mdl.steps_order = [ 'step_1','step_2']
    

    
    
    
    
    # Run analyses
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    mdl.analyse_and_extract(software='ansys_sel', fields=[ 'u', 'sf', 's','eps' ], lstep = ['step_2']) 
    
    
    print('Analysis Finished')
    # # Plot Results
    # # ------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------
    
    # Plot Results for step_2
#    rhino.plot_data(mdl, lstep='step_2', field='uz', cbar_size=1, source='CMMUsermat')
#    rhino.plot_principal_stresses(mdl, step='step_2', shell_layer='top', scale=10**2)
#    rhino.plot_principal_stresses(mdl, step='step_2', shell_layer='bot', scale=10**2)
#    rhino.plot_data(mdl, lstep='step_2', field='sf1', cbar_size=1, source='CMMUsermat')
#    rhino.plot_data(mdl, lstep='step_2', field='sf2', cbar_size=1, source='CMMUsermat')
#    rhino.plot_data(mdl, lstep='step_2', field='sf3', cbar_size=1, source='CMMUsermat')
#    rhino.plot_data(mdl, lstep='step_2', field='sf4', cbar_size=1, source='CMMUsermat')
#    rhino.plot_data(mdl, lstep='step_2', field='sf5', cbar_size=1, source='CMMUsermat')
#    rhino.plot_data(mdl, lstep='step_2', field='sm1', cbar_size=1, source='CMMUsermat')
#    rhino.plot_data(mdl, lstep='step_2', field='sm2', cbar_size=1, source='CMMUsermat')
#    rhino.plot_data(mdl, lstep='step_2', field='sm2', cbar_size=1, source='CMMUsermat')
#    rhino.plot_data(mdl, lstep='step_2', field='sm3', cbar_size=1, source='CMMUsermat')

    
    
    # --------------------------Save Analysis Results -----------------------------------
    # Save the Rhino file
    subfolder_path=folder_path+'\\{}_Batch\\{}_{}_CFB'.format(idx_s, idx_s, ID)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
  
    #save structure to a pickle file
    # ATTENTION ich habe das gefuehl, wenn das file schon existiert dann ueberschreibt es das nicht!!S
    save_to_pickle(obj=mdl, ID=ID, idx_s=idx_s, folder_path=subfolder_path, name='structure')
    
    
    
    
    # save results dict to a json file
    res_dict=mdl.results
    save_to_json(save_dict=res_dict,ID=ID,idx_s=idx_s,folder_path=subfolder_path,
                    name='analysisResults')
    

   
#    # save rhino file (with results)
#    filePath=folder_path+'\\{}_{}_geo_analysis.3dm'.format(idx_s,ID)
#    opt = r.FileIO.FileWriteOptions()
#    doc = r.RhinoDoc.ActiveDoc
#    doc.WriteFile(filePath,opt) #note: also saves DocUserText
    
    print('Files saved')

    

