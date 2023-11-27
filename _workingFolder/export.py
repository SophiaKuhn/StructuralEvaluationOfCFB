# Author(s): Compas/Compas FEA Team, Marius  Weber (ETHZ, HSLU T&A)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import compas
import csv
import re

if compas.RHINO:
    import rhinoscriptsyntax as rs
    import Rhino as r
    import scriptcontext as sc



def read_csv_to_dict(csv_file_path):
    result_dict = {}
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Read the first row as the header

        # Initialize the dictionary with empty lists for each column
        for column_name in header:
            result_dict[column_name] = []

        # Read the remaining rows and append values to respective columns
        for row in csvreader:
            for i, value in enumerate(row):
                result_dict[header[i]].append(value)

    return result_dict


def extract_numbers_from_string(input_string):
    # get all int numbers in a string and append then to a list
    # Here, '\d+' matches one or more digits in the string
    matches = re.findall(r'\d+', input_string)
    
    # Convert the matched strings to integers and return the list of numbers
    return [int(match) for match in matches]


def joinMeshes_inShellLayers(shell_layers, active_layer='Default'):

    # Set active doc to the currently open Rhino file
    sc.doc = r.RhinoDoc.ActiveDoc

    # combine multiple meshes in one layer (muessen immer zusammenhaengend sein)
    for layer in shell_layers:
        # make layer the active layer
        layer_path= sc.doc.Layers.FindByFullPath(layer, 0) #set exept_layer to active layer
        sc.doc.Layers.SetCurrentLayerIndex(layer_path, False)
        #get all objects in layer
        obj = rs.ObjectsByLayer(layer)
        # join meshes (if more than one obj in that layer)
        if len(obj)>1: rs.JoinMeshes(obj, True)

    # make active_layer the active layer again
    layer_path= sc.doc.Layers.FindByFullPath(active_layer, 0) #set exept_layer to active layer
    sc.doc.Layers.SetCurrentLayerIndex(layer_path, False)

    

def delete_all(exept_layer="Default"):
    # Set active doc to the currently open Rhino file
    sc.doc = r.RhinoDoc.ActiveDoc
            
    # Delete all elements in the currently activeDoc
    rs.Command("_SelAll", True)
    rs.Command("_Delete",True)
        
    # Delete all layers (except the exept_layer)
    layer= sc.doc.Layers.FindByFullPath(exept_layer, 0) #set exept_layer to active layer
    sc.doc.Layers.SetCurrentLayerIndex(layer, False)
    layerNames=rs.LayerNames() # get names of all layers existing in activeDoc
    for layer in layerNames:
        rs.PurgeLayer(layer) #delets all layers exept the active one (set above)
        
    # Delete the DocumentUserText (by assigning empty value to all keys)
    if rs.GetDocumentUserText()== None:
        print("Document User Text is already empty. No keys exist.")
    else:
        for key in rs.GetDocumentUserText():
            rs.SetDocumentUserText(key,"") # assingning an empty value to a key removes that key
        print("All keys of DocumentUserText were deleated.")


def save_to_pickle(obj, ID, idx_s, folder_path, name = 'unknown'):

    import pickle
    # construct path
    filePath=folder_path+'\\{}_{}_{}.pkl'.format(idx_s,ID,name)
    # Save the object to a pickle file
    with open(filePath, "wb") as pickle_file:
        pickle.dump(obj, pickle_file)

def save_to_json(save_dict,ID, idx_s, folder_path, name = 'unknown'):
    
    import json
    # construct path
    filePath=folder_path+'\\{}_{}_{}.json'.format(idx_s,ID,name)
    # Save the dictionary to a JSON file
    with open(filePath, "w") as json_file:
        json.dump(save_dict, json_file)
        



def export_results(structure,i, step):
    
    #exploring Nodes
    nodes_dict=structure.nodes #gets dicts of all
    #print('node_dict', nodes_dict )
    node0=structure.nodes[0] #gets the 0. node
    #print('single node object', node0 )
    node0_x=[]
    node0_y=[]
    node0_z=[]
    for n in nodes_dict.keys():
        node0_x.append(structure.nodes[n].x) #gets the global x coordinate of node 0.
        node0_y.append(structure.nodes[n].y)
        node0_z.append(structure.nodes[n].z) 
    #print('single node object x coordniate', node0_x )

    nodesxyz=structure.nodes_xyz() #gets all global coordinates of all nodes of structure
    #print('Printing xyz of all nodes',nodesxyz)

    # exploring elements
    elements = structure.elements
    print('Elements',elements)
    elementkeys = structure.elements.keys() 
    print('Printing element keys',elementkeys)
    element0 = structure.elements[0]
    print('Single Element:', element0)
    element0_property=structure.elements[0].element_property
    print('Element property:', element0_property)


    # exploring element_properties
    el_props=structure. element_properties
    print('All el. properties:', el_props)
    el_prop0=structure. element_properties[element0_property]
    print('el_prop0', el_prop0 )

    #exploring results
    results=structure.results #get all analysis results
    print('Results:', results)

    # ux
    results_nodal_ux=structure.results[step]['nodal']['ux']
    print('Nodal Results:', results_nodal_ux)

    results_nodal_uy=structure.results[step]['nodal']['uy']
    results_nodal_uz=structure.results[step]['nodal']['uz']

    # import csv
    # # Specify the file path
    # file_path = 'C:\Temp\Data\{}_ux.csv'.format(i)

    # # Open the file in write mode
    # with open(file_path, 'w') as csv_file:
    #     writer = csv.writer(csv_file)

    #     # Write the header
    #     writer.writerow(['Key', 'Value'])

    #     # Write the data
    #     for key, value in results_nodal_ux.items():
    #         writer.writerow([key, value])


    # # Specify the file path
    # file_path = 'C:\Temp\Data\{}_uy.csv'.format(i)

    # # Open the file in write mode
    # with open(file_path, 'w') as csv_file:
    #     writer = csv.writer(csv_file)

    #     # Write the header
    #     writer.writerow(['Key', 'Value'])

    #     # Write the data
    #     for key, value in results_nodal_uy.items():
    #         writer.writerow([key, value])


    # # Specify the file path
    # file_path = 'C:\Temp\Data\{}_uz.csv'.format(i)

    # # Open the file in write mode
    # with open(file_path, 'w') as csv_file:
    #     writer = csv.writer(csv_file)

    #     # Write the header
    #     writer.writerow(['Key', 'Value'])

    #     # Write the data
    #     for key, value in results_nodal_uz.items():
    #         writer.writerow([key, value])


    # # save coordinates (list format), in a csv file


    # # # Specify the file path
    # file_path = 'C:\Temp\Data\{}_coorx.csv'.format(i)
    # # Open the file in write mode
    # with open(file_path, 'w') as csv_file:
    #     writer = csv.writer(csv_file)

    #     # Write each list element as a separate row
    #     for item in node0_x:
    #         writer.writerow([item])

    # # # Specify the file path
    # file_path = 'C:\Temp\Data\{}_coory.csv'.format(i)
    # # Open the file in write mode
    # with open(file_path, 'w') as csv_file:
    #     writer = csv.writer(csv_file)

    #     # Write each list element as a separate row
    #     for item in node0_y:
    #         writer.writerow([item])

    # # # Specify the file path
    # file_path = 'C:\Temp\Data\{}_coorz.csv'.format(i)
    # # Open the file in write mode
    # with open(file_path, 'w') as csv_file:
    #     writer = csv.writer(csv_file)

    #     # Write each list element as a separate row
    #     for item in node0_z:
    #         writer.writerow([item])


    print('Data saved to CSV successfully.')


    return None


# ==============================================================================
# Debugging
# ==============================================================================


if __name__ == "__main__":

    pass
