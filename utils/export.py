# Author(s): Compas/Compas FEA Team, Marius  Weber (ETHZ, HSLU T&A)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import compas
# if compas.RHINO:
#     from compas_rhino.geometry import RhinoMesh

# from compas.datastructures.mesh import Mesh
# from compas.datastructures import Network
# from compas.geometry import Frame, Transformation, Vector
# from compas.geometry import add_vectors
# from compas.geometry import cross_vectors
# from compas.geometry import length_vector
# from compas.geometry import scale_vector
# from compas.geometry import subtract_vectors
# from compas_fea.structure import Structure
# from time import time

# from compas_fea.utilities import colorbar
# from compas_fea.utilities import extrude_mesh
# from compas_fea.utilities import network_order
# from compas.rpc import Proxy


# if not compas.IPY:
#     from compas_fea.utilities import meshing
#     from compas_fea.utilities import functions

# else:
#     from compas.rpc import Proxy
#     functions = Proxy('compas_fea.utilities.functions')
#     meshing = Proxy('compas_fea.utilities.meshing')

if compas.RHINO:
    import rhinoscriptsyntax as rs





def export_results(structure):
    
    nodes=structure.nodes_xyz()
    print('Printing nodes',nodes)

    elementkeys = structure.elements.keys()
    print('Printing el keys',elementkeys)

    return None



# ==============================================================================
# Debugging
# ==============================================================================


if __name__ == "__main__":

    pass
