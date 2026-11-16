# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 10:35:35 2025

@author: Matteo Iurato
"""

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
    

    # Imported here so that Polhemus-only users do not need ezc3d installed
    import ezc3d

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
    


# ---------------------------------------------------------------------------
# Least-squares segment frames
# ---------------------------------------------------------------------------
# build_reference_frames() above constructs each frame directly from three markers,
# frame by frame. Its second axis runs along (marker3 - marker2); when those two
# markers are close together (35 mm for the condyle markers of the servo-knee rig)
# every bit of marker noise is amplified into orientation noise by that short
# baseline, and markers beyond the first three are not used at all.
#
# The functions below instead fit ONE rigid body to all the markers of a segment at
# every frame (Soderkvist & Wedin 1993 / Kabsch), then attach the same anatomical
# frame definition to the fitted body. The frame keeps the same meaning; it is just
# estimated from all the information available. The fit residual doubles as a
# quality check: it spikes when a marker is mis-reconstructed.

def marker_array(marker_data, names):
    """Stack named markers into an array of shape (n_frames, n_markers, 3)."""
    return np.stack([np.c_[marker_data[n]['x'], marker_data[n]['y'], marker_data[n]['z']]
                     for n in names], axis=1).astype(float)


def three_marker_frame(m1, m2, m3):
    """Frame convention of build_reference_frames(), vectorised.

    Origin at the midpoint of m2 and m3; rotation matrix with columns [k, i, j] where
    i points from the midpoint to m3, j is normal to the marker plane, k = i x j.
    Inputs have shape (..., 3); returns R (..., 3, 3) and origin (..., 3).
    """
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    mid = (m2 + m3) / 2
    h = unit(m1 - mid)
    i = unit(m3 - mid)
    j = unit(np.cross(h, i))
    k = unit(np.cross(i, j))
    return np.stack([k, i, j], axis=-1), mid


def rigid_body_fit(X, iterations=3):
    """Least-squares rigid motion of a marker cluster.

    X has shape (n_frames, n_markers, 3) with at least three non-collinear markers.
    Returns
      R         (n_frames, 3, 3) rotation of the cluster template into the lab
      p         (n_frames, 3)    position of the template origin (cluster centroid)
      template  (n_markers, 3)   mean cluster shape, centred on its centroid
      residual  (n_frames,)      RMS distance between fitted and measured markers

    The template is the median shape over the whole recording (generalised Procrustes
    with a median in place of the mean), so it is far less noisy than the markers of any
    single frame and is not dragged along by a marker that sits out of place for part
    of the recording.
    """
    X = np.asarray(X, float)
    if np.isnan(X).any():
        bad = int(np.isnan(X).any(axis=(1, 2)).sum())
        raise ValueError(f"{bad} frames have missing markers; fill or cut the gaps first")
    centroid = X.mean(axis=1)
    Xc = X - centroid[:, None, :]
    template = Xc[0].copy()
    for _ in range(max(iterations, 1)):
        H = np.einsum('mi,nmj->nij', template, Xc)
        U, _, Vt = np.linalg.svd(H)
        V = np.swapaxes(Vt, 1, 2)
        det = np.linalg.det(V @ np.swapaxes(U, 1, 2))
        D = np.tile(np.eye(3), (len(X), 1, 1))
        D[:, 2, 2] = det                                  # guard against reflections
        R = V @ D @ np.swapaxes(U, 1, 2)
        template = np.median(np.einsum('nji,nmj->nmi', R, Xc), axis=0)   # median of R^T x
        template -= template.mean(axis=0)
    fitted = np.einsum('nij,mj->nmi', R, template)
    residual = np.sqrt(np.mean(np.sum((fitted - Xc) ** 2, axis=2), axis=1))
    return R, centroid, template, residual


def build_reference_frames_lsq(marker_data, markers_j1, markers_j2):
    """Segment frames from a least-squares rigid-body fit of each marker cluster.

    markers_j1, markers_j2 : lists of marker names for the two segments, three or
        more each. The FIRST THREE of each list define the frame exactly as
        build_reference_frames() does (same origin and axes); any further markers
        only make the estimate more precise.

    Returns J1_R, J1_T, J2_R, J2_T, loc1, loc2 like build_reference_frames(), plus a
    seventh value: a dict with the fit residual of each segment per frame
    ('residual_j1', 'residual_j2', in the units of the markers).
    """
    out = []
    residuals = {}
    for key, names in (('j1', markers_j1), ('j2', markers_j2)):
        if len(names) < 3:
            raise ValueError("each segment needs at least three markers")
        R, p, template, residual = rigid_body_fit(marker_array(marker_data, names))
        R_local, o_local = three_marker_frame(template[0], template[1], template[2])
        R_seg = R @ R_local
        origin = p + np.einsum('nij,j->ni', R, o_local)
        T = np.tile(np.eye(4), (len(R), 1, 1))
        T[:, :3, :3] = R_seg
        T[:, :3, 3] = origin
        out.append((R_seg, T, origin))
        residuals['residual_' + key] = residual
    (R1, T1, loc1), (R2, T2, loc2) = out
    return R1, T1, R2, T2, loc1, loc2, residuals
