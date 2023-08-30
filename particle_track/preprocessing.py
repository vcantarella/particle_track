# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 13:44:22 2023
@author: vcantarella

TODO: 
    - make a more general prepare_arrays function, that is able to run diverse flow boundaries and assing IFACE or
    termination cell according to user specified definitions or instructions on the MODFLOW files.
    - use flopy implementation of get_structured face flows 
"""

import numpy as np
import os
import flopy
from .structured_face_flows import get_structured_faceflows

def prepare_arrays(gwfmodel, model_directory):
    '''
    EDIT this to make sure you add the Boundary conditions in the right faces!!
    Parameters
    ----------
    gwfmodel : TYPE
        Flopy groundwater flow model. Already solved
    model_directory: str
        relative model directory location (Where it will search for modflow files)

    Returns
    -------
    arrays to be used in the particle tracking scheme:
        - 
    
    '''
    ## Reading all the data:
    grid = gwfmodel.modelgrid
    heads = gwfmodel.output.head().get_data()
    budget = gwfmodel.output.budget()
    flow_ja_face = budget.get_data(text = 'FLOW-JA-FACE')[0]
    try:
        wel_flow = budget.get_data(text = 'WEL')
        wel_flow = wel_flow[0]
        wel_nodes = wel_flow['node']-1 #get lrc needs 0 array indices
        wel_index = grid.get_lrc(wel_nodes) #
        wel_q = wel_flow['q']
        wel_iface = wel_flow['IFACE'].astype(np.int16)
    except:
        wel_flow = None
        wel_index = None
        wel_iface = None
        wel_q = None
    try:
        chd_flow = budget.get_data(text = 'CHD')
        chd_flow = chd_flow[0]
        chd_nodes = chd_flow['node']-1 #get lrc needs 0 array indices
        chd_iface = chd_flow['IFACE'].astype(np.int16)
        chd_q = chd_flow['q']
        chd_index = grid.get_lrc(chd_nodes)
    except:
        chd_flow = None
        chd_index = None
        chd_iface = None
        chd_q = None
    
    ## Indicator that says where the particle should be terminated here:
    termination = np.zeros_like(heads).astype(np.int16)
    # for ind in chd_index:
    #     termination[ind] = 1
    # if (wel_flow is not None):
    #     for ind in wel_index:
    #         termination[ind] = 1
    
    structured_face_flows = get_structured_faceflows(flow_ja_face, grb_file=os.path.join(model_directory,gwfmodel.name+'.dis.grb'), verbose=True)
    structured_face_flows = np.stack(structured_face_flows)
    saturated_thickness = grid.saturated_thick(heads)
    
    ## Building dataset. Here we assume the left CHD goes inflow and the right CHD goes outflow.
    
    #Getting all faces for all cells:
    all_faces = np.zeros((6, structured_face_flows.shape[1], structured_face_flows.shape[2], structured_face_flows.shape[3]))
    all_faces[1] = structured_face_flows[0] # right face (MODFLOW convention) (right x)
    all_faces[2] = (-1)*structured_face_flows[1] # front face (MODFLOW convention) (lower y direction)
    all_faces[4] = (-1)*structured_face_flows[2] # bottom face (MODFLOW convention) (lower z direction)

    all_faces[0,:,:,1:] = structured_face_flows[0,:,:,:-1]#left x face
    all_faces[3,:,1:,:] = (-1)*structured_face_flows[1,:,:-1,:] #top y face
    all_faces[5,1:,:,:] = (-1)*structured_face_flows[2,:-1,:,:] #upper z face

    if chd_flow is not None:
        i = 0
        for ind in chd_index:
            iface = chd_iface[i].astype(np.int16)
            if iface == 1: #left face:
                ind = (0,)+ind
                all_faces[ind] = all_faces[ind]+ chd_q[i]
            elif iface == 2:
                ind = (1,)+ind
                all_faces[ind] = all_faces[ind] - chd_q[i]
            elif iface == 6:
                ind = (5,)+ind
                all_faces[ind] = all_faces[ind] - chd_q[i]
            else:
                i+=1
                continue
            i+=1
        
    if (wel_flow is not None):
        i = 0
        for ind in wel_index:
            iface = wel_iface[i]
            if iface == 1: #left face:
                ind = (0,)+ind
                all_faces[ind] = all_faces[ind] + wel_q[i]
            elif iface == 2:
                ind = (1,)+ind
                all_faces[ind] = all_faces[ind] - wel_q[i]
            elif iface == 6:
                ind = (5,)+ind
                all_faces[ind] = all_faces[ind] - wel_q[i]
            elif iface == 0:
                termination[ind] = 1
            else:
                i+=1
                continue
            i+=1
    
    # all_faces has all the face flows
    ## Calculating cell areas:
    dy = grid.delc # row spacing in the column direction(dy)
    dx = grid.delr #column spacing in the row direction (dx)
    delz = grid.delz

    dy = dy[np.newaxis,:,np.newaxis]
    dx = dx[np.newaxis,np.newaxis,:]

    row_area = dy * saturated_thickness

    col_area = dx * saturated_thickness

    top_area = dx * dy
    
    ## Fixed 30% porosity for now (CHANGE LATER)
    phi = np.ones_like(heads)*0.3
    
    # transforming faces into velocities:
    all_faces[0] = all_faces[0]/row_area/phi
    all_faces[1] = all_faces[1]/row_area/phi
    all_faces[2] = all_faces[2]/col_area/phi
    all_faces[3] = all_faces[3]/col_area/phi
    all_faces[4] = all_faces[4]/top_area/phi
    all_faces[5] = all_faces[5]/top_area/phi
    
    face_velocities = all_faces

    #getting gradients
    gvx = (all_faces[1]-all_faces[0])/dx
    gvy = (all_faces[3]-all_faces[2])/dy
    gvz = (all_faces[5]-all_faces[4])/saturated_thickness
    gvs = np.stack((gvx,gvy,gvz),axis = 0)
    z_uf = grid.botm + saturated_thickness
    z_lf = grid.botm
    xyedges = grid.xyedges
    xedges = xyedges[0]+grid.xoffset
    yedges = xyedges[1]+grid.yoffset

    return xedges,yedges, z_lf, z_uf, gvs, face_velocities, termination