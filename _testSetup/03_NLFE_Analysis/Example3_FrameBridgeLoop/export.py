# Author(s): Sophia Kuhn (ETHZ)

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


def save_to_pickle(obj, folder_path=None, file_name = 'unknown'):

    import pickle
    # construct path
    if folder_path == None:
        filePath= file_name + '.pkl'
    else:
        filePath=folder_path + '\\'+file_name + '.pkl'

    # Save the object to a pickle file
    with open(filePath, "wb") as pickle_file:
        pickle.dump(obj, pickle_file)

def save_to_json(save_dict,folder_path=None, file_name = 'unknown'):
    
    import json
    # construct path
    if folder_path == None:
        filePath=file_name + '.json'
    else:
        filePath=folder_path + '\\'+file_name + '.json'

    # Save the dictionary to a JSON file
    with open(filePath, "w") as json_file:
        json.dump(save_dict, json_file)
        



# ==============================================================================
# Debugging
# ==============================================================================


if __name__ == "__main__":

    pass
