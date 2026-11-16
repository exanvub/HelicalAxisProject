# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 10:35:35 2025

@author: Matteo Iurato
"""

import ezc3d
import numpy as np

def read_data(path):
    """
    

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    c3d : TYPE
        DESCRIPTION.
    marker_data : TYPE
        DESCRIPTION.
    time : TYPE
        DESCRIPTION.

    """
    
    
    # Load a C3D file
    c3d = ezc3d.c3d(path)
    # Access points data (3D motion capture data)
    points = c3d['data']['points']  # shape: (4, n_points, n_frames)
    # The 4th row is usually [X, Y, Z, residual error or 1]
    
    # Access analog data (like force plate data)
    # shape: (n_subframes, n_analog_channels, n_frames)
    analogs = c3d['data']['analogs'] 
    
    # Access parameters
    parameters = c3d['parameters']

    # Print some information
    print("Number of 3D points:", points.shape[1])
    print("Number of frames:", points.shape[2])
    print("Number of analog channels:", analogs.shape[1])
    print("Analog sampling rate:", c3d['header']['analogs']['frame_rate'])
    
    # Example: print coordinates of the first point in the first frame
    x, y, z = points[0, 0, 0], points[1, 0, 0], points[2, 0, 0]
    print(f"First point (frame 0): X={x}, Y={y}, Z={z}")
    
    # Extract marker names
    marker_labels = c3d['parameters']['POINT']['LABELS']['value']
    
    # Print all marker names
    print("Marker labels:")
    for i, label in enumerate(marker_labels):
        print(f"{i}: {label}")

    # Get point data (4, n_markers, n_frames)
    points = c3d['data']['points']
    
    # Dictionary to hold each marker's coordinates
    marker_data = {}
    
    for i, label in enumerate(marker_labels):
        # Get X, Y, Z coordinates across all frames
        x = points[0, i, :]
        y = points[1, i, :]
        z = points[2, i, :]
    
        # Store as a tuple or dict in marker_data
        marker_data[label] = {'x': x, 'y': y, 'z': z}
    
        # Also create variables in the global namespace (optional, not best practice)
        globals()[label] = {'x': x, 'y': y, 'z': z}
       
    #Time vector    
    n_frames = c3d['data']['points'].shape[2]
    frame_rate = c3d['header']['points']['frame_rate']
    time = np.round(np.arange(n_frames)/frame_rate,3)
        
    return c3d, marker_data, time


def build_reference_frames(marker_data, marker1_j1, marker2_j1, marker3_j1, marker1_j2, marker2_j2, marker3_j2):
    """
    

    Parameters
    ----------
    marker_data : TYPE
        DESCRIPTION.
    marker1_j1 : TYPE
        DESCRIPTION.
    marker2_j1 : TYPE
        DESCRIPTION.
    marker3_j1 : TYPE
        DESCRIPTION.
    marker1_j2 : TYPE
        DESCRIPTION.
    marker2_j2 : TYPE
        DESCRIPTION.
    marker3_j2 : TYPE
        DESCRIPTION.

    Returns
    -------
    J1_R : TYPE
        DESCRIPTION.
    J1_T : TYPE
        DESCRIPTION.
    J2_R : TYPE
        DESCRIPTION.
    J2_T : TYPE
        DESCRIPTION.
    loc1 : TYPE
        DESCRIPTION.
    loc2 : TYPE
        DESCRIPTION.

    """
    
    #Compute joint 1 frames
    J1_R = np.zeros((len(marker_data[marker1_j1]['x']), 3, 3))
    J1_T = np.zeros((len(marker_data[marker1_j1]['x']), 4, 4))
    
    loc1 = []
    
    for t in range(len(marker_data[marker1_j1]['x'])):
        m2 = [marker_data[marker2_j1]['x'][t], marker_data[marker2_j1]['y'][t], marker_data[marker2_j1]['z'][t]]
        m3 = [marker_data[marker3_j1]['x'][t], marker_data[marker3_j1]['y'][t], marker_data[marker3_j1]['z'][t]]
        mid_point = (np.array(m2)+np.array(m3))/2


    
        # Primary axis (h-axis)
        m1 = [marker_data[marker1_j1]['x'][t], marker_data[marker1_j1]['y'][t], marker_data[marker1_j1]['z'][t]]
        
        h = (np.array(m1) - mid_point)
        h /= np.linalg.norm(h)
    
        # Secondary axis (i-axis)
        i = (m3 - mid_point)
        i /= np.linalg.norm(i)
    
        # Compute orthogonal axes (j and k)
        j = np.cross(h, i)
        j /= np.linalg.norm(j)
        k = np.cross(i, j)
        k /= np.linalg.norm(k)
        J1_R[t, :, :] = np.c_[k,i,j]
        
        
        # Compute the translation vector (e.g., the origin of the dynamic frame)
        translation_vector = mid_point
    
        # Create the transformation matrix
        transformation_matrix = np.eye(4)  # Initialize a 4x4 identity matrix
        transformation_matrix[:3, :3] = np.c_[k,i,j]  # Set the rotation matrix
        # Set the translation vector
        transformation_matrix[:3, 3] = translation_vector
    
        # Store the transformation matrix
        J1_T[t, :, :] = transformation_matrix
        loc1.append(translation_vector)
    
    #Compute joint 2 frames
    J2_R = np.zeros((len(marker_data[marker1_j2]['x']), 3, 3))
    J2_T = np.zeros((len(marker_data[marker1_j2]['x']), 4, 4))
    
    loc2 = []
    
    for t in range(len(marker_data[marker1_j2]['x'])):
        m2 = [marker_data[marker2_j2]['x'][t], marker_data[marker2_j2]['y'][t], marker_data[marker2_j2]['z'][t]]
        m3 = [marker_data[marker3_j2]['x'][t], marker_data[marker3_j2]['y'][t], marker_data[marker3_j2]['z'][t]]
        mid_point = (np.array(m2)+np.array(m3))/2
    
        # Primary axis (h-axis)
        m1 = [marker_data[marker1_j2]['x'][t], marker_data[marker1_j2]['y'][t], marker_data[marker1_j2]['z'][t]]
        
        h = (np.array(m1) - mid_point)
        h /= np.linalg.norm(h)
    
        # Secondary axis (i-axis)
        i = (m3 - mid_point)
        i /= np.linalg.norm(i)
    
        # Compute orthogonal axes (j and k)
        j = np.cross(h, i)
        j /= np.linalg.norm(j)
        k = np.cross(i, j)
        k /= np.linalg.norm(k)
        J2_R[t, :, :] = np.c_[k,i,j]
        
        # Compute the translation vector (e.g., the origin of the dynamic frame)
        translation_vector = mid_point
    
        # Create the transformation matrix
        transformation_matrix = np.eye(4)  # Initialize a 4x4 identity matrix
        transformation_matrix[:3, :3] = np.c_[k,i,j]  # Set the rotation matrix
        # Set the translation vector
        transformation_matrix[:3, 3] = translation_vector
    
        # Store the transformation matrix
        J2_T[t, :, :] = transformation_matrix
        loc2.append(translation_vector)
    
    return np.array(J1_R), np.array(J1_T), np.array(J2_R), np.array(J2_T), loc1, loc2
    
