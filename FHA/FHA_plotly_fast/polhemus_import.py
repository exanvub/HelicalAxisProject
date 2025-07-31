# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 10:35:21 2025

@author: Matteo Iurato
"""

import numpy as np
import pandas as pd
import time
from scipy.spatial.transform import Rotation as R

def read_data(path):
    """
    

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    q1 : TYPE
        DESCRIPTION.
    q2 : TYPE
        DESCRIPTION.
    loc1 : TYPE
        DESCRIPTION.
    loc2 : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    """
    
    data = pd.read_csv(path)
    
    # Extract relevant data
    # t = []
    # for el in data['Time'].values:
    #     t.append(time.gmtime(el))
    t = np.array(data['Time'].values) - data['Time'].values[0]
    
    q1 = data[['w1', 'x1', 'y1', 'z1']].values
    q2 = data[['w2', 'x2', 'y2', 'z2']].values
    loc1 = data[['loc1_x', 'loc1_y', 'loc1_z']].values
    loc2 = data[['loc2_x', 'loc2_y', 'loc2_z']].values
    
    return q1, q2, loc1, loc2, t

def build_homogeneous_matrix(R, locx, locy, locz):
    """
    

    Parameters
    ----------
    R : TYPE
        DESCRIPTION.
    locx : TYPE
        DESCRIPTION.
    locy : TYPE
        DESCRIPTION.
    locz : TYPE
        DESCRIPTION.

    Returns
    -------
    T : TYPE
        DESCRIPTION.

    """
    
    T = np.matrix([[R[0,0], R[0,1], R[0,2], locx],
                   [R[1,0], R[1,1], R[1,2], locy],
                   [R[2,0], R[2,1], R[2,2], locz],
                   [0     ,      0,      0,    1]
        ])
    
    return T


# Scipy quaternion to rotation matrix convertion
def quaternion_to_rotation_matrix(quaternion):
    """
    

    Parameters
    ----------
    quaternion : TYPE
        DESCRIPTION.

    Returns
    -------
    rotation_matrix : TYPE
        DESCRIPTION.

    """
    
    quat = np.roll(quaternion, -1) #rearrange quaternion components as x,y,z,w for scipy
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()
    return rotation_matrix



def quaternions_to_matrices(q1, q2, loc1, loc2):
    """
    

    Parameters
    ----------
    q1 : TYPE
        DESCRIPTION.
    q2 : TYPE
        DESCRIPTION.
    loc1 : TYPE
        DESCRIPTION.
    loc2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    R1 = []
    R2 = []
    for i in range(len(q1)):
        
        # Use this if you rely on scipy conversion
        R1.append(quaternion_to_rotation_matrix(q1[i]))
        R2.append(quaternion_to_rotation_matrix(q2[i]))

    def build_homogeneous_matrix(R, locx, locy, locz):
        T = np.matrix([[R[0,0], R[0,1], R[0,2], locx],
                       [R[1,0], R[1,1], R[1,2], locy],
                       [R[2,0], R[2,1], R[2,2], locz],
                       [0     ,      0,      0,    1]
            ])
        
        return T
        
    T1 = []
    T2 = []
    for i in range(len(R1)):
        T1.append(build_homogeneous_matrix(R1[i], loc1[i][0], loc1[i][1], loc1[i][2]))
        T2.append(build_homogeneous_matrix(R2[i], loc2[i][0], loc2[i][1], loc2[i][2]))
        
    T1 = np.array(T1)
    T2 = np.array(T2)
    
    return R1, T1, R2, T2


    



