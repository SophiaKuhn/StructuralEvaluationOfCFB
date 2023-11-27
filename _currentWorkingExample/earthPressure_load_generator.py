 # Author(s): Sophia Kuhn (ETH Zurich)

import os 
import math as m
import rhinoscriptsyntax as rs
import scriptcontext
import System.Guid, System.Drawing.Color
from compas_fea.structure import Structure
from compas_fea.cad import rhino
from compas_fea.structure import AreaLoad
from strucenglib.prepost_functions import area_load_generator_elements



def earthPressure_backfill_generator(structure, elements, h_w, t_p, h_G, gamma_E, phi_k, verbalise=True):
    
    '''
    This function calculates the earth pressure area load magnitude that results from the backfill and generates the correspondin area load.

    Parameters
    ----------
    structure: structureObject
        the object representing the structure to be analysed
    elements: List[str]
        List of the names of the layers, which contain the elements that should be loaded   
    h_w : float
        Wall hight [mm]
    t_p : float
        Deck Slab Thickness [mm]
    h_G : float
        Gravel layer hight [mm]
    gamma_E : float
        sp. Weight [N/mm3] (e.g. Verdichteter Schotter 0.000020 N/mm3 )
    phi_k: int
        ..[Degree] (e.g. 30)
    verbalise: bool
        Defining weather to verbalise the calculation 


    Returns
    ----------
    List[str]
        List of load names, which were generated within this function: ['earthPressure_backfill']

    Limitations
    -------------
    - NO waterpressure is considered (for low water levels), 
    - constant gamma is considered (for gravel and backfill), so no layerd ground
    '''
    

    # Basic definitions
    #-------------------------------------------------------

    #calc. Ko
    Ko = 1 - m.sin(phi_k) 

    # calculate hight over which the earth pressure is active onto the structure
    # (only correct for with offsetmodelling and with MPCs and voute)
    h_ep = h_w  + t_p #[mm]

    # celculate earth pressure
    # at top of slab deck
    h_ep_o=h_G #[mm]
    p_t = Ko * gamma_E * h_ep_o #[N/mm2]
    # at foundation hight
    h_ep_u = h_ep + h_G #[mm]
    p_f = Ko * gamma_E * h_ep_u #[N/mm2] 

    # calculate resulting force  (area of earth pressure)
    ep_r = p_t * h_ep + 1/2 *p_f * h_ep #[N/mm] # TODO muss hier nicht p_f p_t stehen

    # dirtribute uniformily on whole elset hight
    q_ep_r=ep_r/ h_w  #[N/mm2]

    # verbalise
    if verbalise:
        print('The earth pressure resulting from the backfill is calculated to be: ', q_ep_r, ' N/mm2 ;',q_ep_r*1000, ' kN/m2'  )

    structure.add(AreaLoad(name='earthPressure_backfill', elements=elements, x=0,y=-q_ep_r,z=0, axes ='local')) 

    # return q_er_r 
    return ['earthPressure_backfill']


def earthPressure_liveload_generator(structure, s, h_w, t_p, phi_k, verbalise=True):
    
    '''
    This function calculates the earth pressure area load magnitude that results from the live load of a track
    and generates the correspondin area load.

    Parameters
    ----------
    structure: structureObject
        object representing the structure to be analysed
    s : float
        Distance between origin and middle axis of the track 
    h_w : float
        Wall hight [mm]
    t_p : float
        Deck Slab Thickness [mm]
    phi_k: int
        ..[Degree] (e.g. 30)
    verbalise: bool
        Defining weather to verbalise the calculation 


    Returns
    ----------
    List[str]
        List of load names, which were generated within this function: ['earthPressure_backfill']

    Limitations/Specification
    -------------
    - Has to be used after the the Normalspurbahnverkehr_load_generator function was executed (as this function uses the created Mittelachse)
    # TODO if Mittelachse noch nicht exist create one, coder von Normalspur function hier einfuegen?
    '''

    # Create layer
    #-------------------------------------------------------
    # define layer name ( and later name of area load)
    layer_name = "EarthPressure_liveLoad_area"

    # create the new layer
    if rs.IsLayer(layer_name):
        rs.PurgeLayer(layer_name)
        scriptcontext.doc.Layers.Add(layer_name, System.Drawing.Color.Green)
    else:
        scriptcontext.doc.Layers.Add(layer_name, System.Drawing.Color.Green)
    
    # set layer to current active layer
    rs.CurrentLayer(layer_name)

    # Generate polyline of area load
    #-------------------------------------------------------
    # Startpunkt der Mittelachse (Annahme: Globaler Nullpunkt immer bei x=0,y=0,z=0)
    point_start_x=s
    point_start_y=0

    #calculation of corner point coordinates in wall 1
    # y is always 0
    ep_width=3800 #[mm] #TODO why 3800, not 3000, should be varied?, or maybe gesamte b_GL_strich laenge aus Normal...function nehmen?
    x_A= point_start_x +ep_width/2
    z_A = -t_p
    x_B= point_start_x +ep_width/2
    z_B = -h_w-t_p
    x_C= point_start_x -ep_width/2
    z_C = -h_w-t_p
    x_D= point_start_x -ep_width/2
    z_D = -t_p

    # creation of polyline, and add to active layer
    rs.AddPolyline([(x_A,0,z_A),(x_B,0,z_B),(x_C,0,z_C),(x_D,0,z_D), (x_A,0,z_A)])


    # calculate coordinates of corner points in wall 2
    # TODO heisst "einsitig" check if it should be applid to wall 1 instead of wall 2
    # TODO for oter side: select mittelachse (von Gleis aufbringung)




    # Calculate magnitude and add area load
    #-------------------------------------------------------
    # calculate magnitude of area load
    q = 52 #[N/mm2] #TODO check where this value comes from and weather it has to be varied
    Ko = 1-m.sin(phi_k)
    ep = Ko * q #[N/mm2]

    # determine which elements are loaded
    loaded_element_numbers=area_load_generator_elements(structure, layer_name)

    # add load
    load_name='earthPressure_liveLoad'
    structure.add(AreaLoad(load_name, elements=loaded_element_numbers,x=0,y=-ep,z=0,axes ='local'))

    # set default layer back to active layer
    rs.CurrentLayer('Default')

    # verbalise
    if verbalise:
        print('The earth pressure resulting from the live load is calculated to be: ', ep, ' N/mm2 ;',ep*1000, ' kN/m2'  )

    #return of name of generated area load
    return [load_name]


