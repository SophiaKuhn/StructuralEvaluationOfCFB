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



def earthPressure_gravel_generator(structure, elements,h_G, gamma_E, phi_k, gamma_G=1, verbalise=True):

    '''
    This function calculates the earth pressure area load magnitude that results from the layers of soil that lie above the slab deck.
    And then this function adds this caluclated load value as a load to the structure object.

    Parameters
    ----------
    structure: structureObject
        the object representing the structure to be analysed
    elements: List[str]
        List of the names of the layers, which contain the elements that should be loaded   
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
    - constant gamma_E is considered (for gravel and backfill), so no layerd ground
    '''


    #calc. Ko
    Ko = 1 - m.sin(m.radians(phi_k))

    # calc eath pressure at slab deck surface (resulting from gravel layer)
    p = gamma_G*(Ko * gamma_E * h_G) #[N/mm2]  (gamma_E: [N/mm3])

    # verbalise
    if verbalise:
        print('The earth pressure resulting from the gravel layer is calculated to be: ', p, ' N/mm2 ;',p*1000, ' kN/m2'  )

    #add load to structur object
    structure.add(AreaLoad(name='earthPressure_gravel', elements=elements, x=0,y=0,z=-p, axes ='local')) 

    #return the name of the load that was saved to the structure object
    return ['earthPressure_gravel']






def earthPressure_backfill_generator(structure, elements, h_w, t_p, h_G, gamma_E, phi_k, gamma_G=1, verbalise=True):
    
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
    Ko = 1 - m.sin(m.radians(phi_k)) 

    # calculate hight over which the earth pressure is active onto the structure
    # (only correct for with offsetmodelling and with MPCs and voute)
    h_ep = h_w  + t_p #[mm]

    # celculate earth pressure
    # at top of slab deck (for the sqare of the earth pressure)
    p_t = Ko * gamma_E * h_G #[N/mm2]

    # at foundation hight (for the triangle of the force pressure)
    p_f = Ko * gamma_E * h_ep #[N/mm2] 


    # calculate resulting force  (area of earth pressure)
    R_sqare=gamma_G* p_t * h_ep  #[N/mm] 
    R_triangle = gamma_G*1/2 *p_f * h_ep #[N/mm] 
    R_tot=R_sqare+R_triangle #[N/mm] 


    # dirtribute uniformily on whole elset hight
    q_ep_r=R_tot/ h_w  #[N/mm2]

    # verbalise
    if verbalise:
        print('The earth pressure resulting from the backfill is calculated to be: ', q_ep_r, ' N/mm2 ;',q_ep_r*1000, ' kN/m2' )

    structure.add(AreaLoad(name='earthPressure_backfill', elements=elements, x=0,y=-q_ep_r,z=0, axes ='local')) 

    # return q_er_r 
    return ['earthPressure_backfill']


def earthPressure_liveload_generator(structure, s, h_w, t_p, phi_k,gamma_Q=1, verbalise=True):
    
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
    - So far the earth pressure live load is only applied to one side (at x=0), so far this function is not able to apply earth pressure on both wall sides
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
    ep_width=3800 #[mm] # From PAIngB B1.3 Page 72 (SBB specific norm)
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
    # TODO Erddruck infolge vertikaler Bahnverkehrslasten ist einseitig oder beidseitig des Rahmens anzunehmen. (laut PAIngB B1.3 Page 72)
    # TODO bis jetzt nur einseitig aufgebracht (zweiseitig ist bis jetzt nicht moeglich mit dieser funktion)




    # Calculate magnitude and add area load
    #-------------------------------------------------------
    # calculate magnitude of area load
    q = 0.052 #[N/mm2] = 52[kN/m2] From PAIngB B1.3 Page 72 (SBB specific norm)
    # z = 700 [mm] #From PAIngB B1.3 Page 72
    #TODO checken ob das so richtig ist mit dem If sentence (vgl bericht Hero)
    # if h_g > z:
    #     z=h_g
    Ko = 1-m.sin(m.radians(phi_k))
    p = gamma_Q*(Ko * q) #[N/mm2]

    # determine which elements are loaded
    loaded_element_numbers=area_load_generator_elements(structure, layer_name)

    # add load
    load_name='earthPressure_liveLoad'
    structure.add(AreaLoad(load_name, elements=loaded_element_numbers,x=0,y=-p,z=0,axes ='local'))

    # set default layer back to active layer
    rs.CurrentLayer('Default')

    # verbalise
    if verbalise:
        print('The earth pressure resulting from the live load is calculated to be: ', p, ' N/mm2 ;',p*1000, ' kN/m2'  )

    #return of name of generated area load
    return [load_name]


