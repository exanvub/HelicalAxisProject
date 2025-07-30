import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R
import math
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import html
from dash import dcc

def decompose_homogeneous_matrix(H):
    """Original function for single matrix decomposition."""
    Horig = np.array(H)
    R = Horig[0:3, 0:3]
    v = Horig[0:3, 3].transpose()
    return R, v
    
### Method 1
def calculate_FHA_vectorized(T1, T2):
    """
    Vectorized calculation of Finite Helical Axis (FHA) parameters.
    Takes two stacks of homogeneous matrices (N, 4, 4) and returns FHA parameters for each pair.
    """
    # Ensure inputs are numpy arrays
    T1 = np.array(T1)
    T2 = np.array(T2)

    # Vectorized calculation of relative homogeneous transform H
    # H = T2 * inv(T1)
    H = T2 @ np.linalg.inv(T1)

    # Vectorized decomposition of H into rotation (ROT) and translation (v)
    ROT = H[:, 0:3, 0:3]
    v = H[:, 0:3, 3]

    # Vectorized calculation of sin(phi) and cos(phi)
    # Using elements of the rotation matrix
    sinPhi = 0.5 * np.sqrt(
        (ROT[:, 2, 1] - ROT[:, 1, 2])**2 +
        (ROT[:, 0, 2] - ROT[:, 2, 0])**2 +
        (ROT[:, 1, 0] - ROT[:, 0, 1])**2
    )
    cosPhi = 0.5 * (np.trace(ROT, axis1=1, axis2=2) - 1)

    # Vectorized calculation of phi, handling the quadrant ambiguity
    # Use np.where for conditional calculation based on sinPhi
    phi = np.where(sinPhi <= (np.sqrt(2) / 2), np.arcsin(sinPhi), np.arccos(cosPhi))

    # Add a small epsilon to avoid division by zero for pure translations (sinPhi=0)
    epsilon = 1e-12
    
    # Vectorized calculation of the axis vector 'n'
    n_unnormalized = np.stack([
        ROT[:, 2, 1] - ROT[:, 1, 2],
        ROT[:, 0, 2] - ROT[:, 2, 0],
        ROT[:, 1, 0] - ROT[:, 0, 1]
    ], axis=-1)
    
    n = n_unnormalized / (2 * sinPhi[:, np.newaxis] + epsilon)

    # Vectorized calculation of translation along the axis 't'
    # einsum is used for batched dot product: sum over products of elements for each vector pair
    t = np.einsum('ij,ij->i', n, v)

    # Vectorized calculation of the location of the FHA 's'
    # Re-estimate sin and cos from the calculated phi for the formula
    s_cos_phi = np.cos(phi)
    s_sin_phi = np.sin(phi)
    
    # Using vector triple product identity and half-angle formulas for 's'
    # This avoids loops and performs calculations on the entire arrays.
    term1 = np.cross(-0.5 * n, np.cross(n, v))
    factor = s_sin_phi / (2 * (1 - s_cos_phi) + epsilon)
    term2 = np.cross(factor[:, np.newaxis] * n, v)
    s = term1 + term2
    
    return phi, n, t, s

def activate_method(method_type, t, Trel, loc1, loc2, all_hax, all_angles, all_svec, all_d, all_translation_1_list, all_translation_2_list, step):
    hax, ang, svec, d, translation_1_list, translation_2_list = [], [], [], [], [], []
    ind_incr, ind_step = [], []
    
    if method_type != 'all_FHA':    
        time_diff = []

    time_incr, ang_incr = [], []
    incrAng, indCount = [], []

    # This part remains sequential as it determines the indices for subsequent calculations
    if method_type in ['step_angle', 'incremental_angle', 'incremental_time']:
        angSum = 0
        incrAng = np.cumsum(all_angles)

    if method_type == 'all_FHA':
        return all_hax, all_angles, all_svec, all_d, all_translation_1_list, all_translation_2_list, ind_incr, ind_step, incrAng, t[:-1], all_angles

    elif method_type == 'incremental_time':
        # Define start and end matrices for vectorized calculation
        T1_in = Trel[:-step]
        T2_in = Trel[step:]
        phi, n, tran, s = calculate_FHA_vectorized(T1_in, T2_in)
        
        hax, ang, svec, d = n, phi, s, tran
        translation_1_list = loc1[:-step]
        translation_2_list = loc2[:-step]
        
        time_incr = t[:-step]
        ang_incr = incrAng[:-step]
        time_diff = t[step:] - t[:-step]

    elif method_type == 'step_angle':
        # This loop is inherently sequential and finds the indices based on cumulative angle
        angSum = 0
        ind_step = [0]
        for i in range(len(all_angles)):
            angSum += all_angles[i]
            if angSum > step:
                ind_step.append(i)
                angSum = 0
        ind_step = np.array(ind_step)

        # Vectorized calculation using the determined indices
        T1_in = Trel[ind_step[:-1]]
        T2_in = Trel[ind_step[1:]]
        phi, n, tran, s = calculate_FHA_vectorized(T1_in, T2_in)

        hax, ang, svec, d = n, phi, s, tran
        translation_1_list = loc1[ind_step[:-1]]
        translation_2_list = loc2[ind_step[:-1]]
        
        time_incr = t[ind_step[:-1]]
        ang_incr = [incrAng[i] for i in ind_step[1:]] # Cumulative angle at the end of the step
        time_diff = t[ind_step[1:]] - t[ind_step[:-1]]
    
    elif method_type == 'incremental_angle':
        angSum = 0
        ind_incr = []
        for i in range(len(all_angles) - 1):
            angSum = 0
            for j in range(i, len(all_angles)):
                angSum += all_angles[j]
                if angSum > step:
                    ind_incr.append([i, j])
                    break
        
        if not ind_incr: # Handle case where no intervals meet the criteria
             return [], [], [], [], [], [], [], [], [], [], []

        start_indices = np.array([p[0] for p in ind_incr])
        end_indices = np.array([p[1] for p in ind_incr])

        # Vectorized calculation using the determined index pairs
        T1_in = Trel[start_indices]
        T2_in = Trel[end_indices]
        phi, n, tran, s = calculate_FHA_vectorized(T1_in, T2_in)
        
        hax, ang, svec, d = n, phi, s, tran
        translation_1_list = loc1[start_indices]
        translation_2_list = loc2[start_indices]

        time_incr = t[start_indices]
        # Calculate cumulative angle for each found interval
        ang_incr = [np.sum(all_angles[pair[0]:pair[1]+1]) for pair in ind_incr]
        time_diff = t[end_indices] - t[start_indices]

    else:
        raise ValueError("Invalid method type. Choose from 'all_FHA', 'incremental_time', 'step_angle', or 'incremental_angle'.")
    
    return hax, ang, svec, d, translation_1_list, translation_2_list, ind_incr, ind_step, incrAng, time_incr, ang_incr


def generate_FHA(method_type, t, cut1, cut2, step, nn, R1, R2, T1, T2, loc1, loc2):
    ###### Cut data ######
    R1, R2 = R1[cut1:cut2], R2[cut1:cut2]
    T1, T2 = T1[cut1:cut2], T2[cut1:cut2]
    loc1, loc2 = loc1[cut1:cut2], loc2[cut1:cut2]
    t = t[cut1:cut2]

    # Vectorized calculation of Trel = inv(T1) @ T2
    T1inv = np.linalg.inv(T1)
    Trel = T1inv @ T2
    
    # Calculate all instantaneous FHAs in a single vectorized call
    all_angles_rad, all_hax, all_d, all_svec = calculate_FHA_vectorized(Trel[:-1], Trel[1:])
    all_angles = np.degrees(all_angles_rad)
    
    all_translation_1_list = loc1[:-1]
    all_translation_2_list = loc2[:-1]
    
    time_diff = t[1:] - t[:-1]
        
    # Activate the chosen method to get the final FHA parameters
    hax, ang, svec, d, translation_1_list, translation_2_list, ind_incr, ind_step, incrAng, time_incr, ang_incr = activate_method(
        method_type, t, Trel, loc1, loc2, all_hax, all_angles, all_svec, all_d, 
        all_translation_1_list, all_translation_2_list, step
    )

    return hax, ang, svec, d, translation_1_list, translation_2_list, time_diff, time_incr, ang_incr, all_angles, ind_incr, ind_step, t


def calculate_intersection_vectorized(fha_origins, fha_directions, plane_point, plane_normal):
    """ Vectorized calculation of line-plane intersections. """
    epsilon = 1e-12
    
    # Normalize all direction vectors
    fha_directions_norm = fha_directions / (np.linalg.norm(fha_directions, axis=1, keepdims=True) + epsilon)
    plane_normal_norm = plane_normal / (np.linalg.norm(plane_normal) + epsilon)

    # Batched dot product for numerator and denominator
    numerator = np.einsum('ij,j->i', plane_point - fha_origins, plane_normal_norm)
    denominator = np.einsum('ij,j->i', fha_directions_norm, plane_normal_norm)

    # Find where the line is not parallel to the plane
    valid_indices = np.abs(denominator) > epsilon
    
    # Initialize intersection points array
    intersections = np.full_like(fha_origins, np.nan)
    
    if np.any(valid_indices):
        t = numerator[valid_indices] / denominator[valid_indices]
        # Calculate intersection points only for non-parallel lines
        intersections[valid_indices] = fha_origins[valid_indices] + t[:, np.newaxis] * fha_directions_norm[valid_indices]
        
    return intersections

def plotly_visualize_FHA(isPolhemus, T1, T2, method_type, hax, svec, d, ind_incr, ind_step, nn, translation_1_list, translation_2_list, t, time_incr, ang_incr, step, all_angles, time_diff):

    # Ensure inputs are numpy arrays for vectorization
    hax, svec, d = np.array(hax), np.array(svec), np.array(d)
    
    if hax.shape[0] == 0:
        print("No FHA data to visualize.")
        return go.Figure()

    # Decompose T1 once to get all R1 matrices
    R1 = np.array([decompose_homogeneous_matrix(m)[0] for m in T1])
    R2 = np.array([decompose_homogeneous_matrix(m)[0] for m in T2])

    # Initialize euler_angles as empty array (will be populated for some methods)
    euler_angles = np.array([])

    # Select the correct transformation matrices based on the method
    if method_type == 'all_FHA':
        R1_transform = R1[:-1]
        T1_transform = T1[:-1]

    elif method_type == 'incremental_time':
        R1_transform = R1[:-step]
        T1_transform = T1[:-step]

    elif method_type == 'step_angle':
        indices = np.array(ind_step)[:-1]
        R1_transform = R1[indices]
        T1_transform = T1[indices]

        R2_transform = R2[np.array(ind_step)[1:]]
        euler_angles = np.array([R.from_matrix(r).as_euler('xyz', degrees=True) for r in R2_transform])

    elif method_type == 'incremental_angle':
        start_indices = np.array([p[0] for p in ind_incr])
        R1_transform = R1[start_indices]
        T1_transform = T1[start_indices]

        R2_transform = R2[np.array([p[1] for p in ind_incr])]
        euler_angles = np.array([R.from_matrix(r).as_euler('xyz', degrees=True) for r in R2_transform])


    # Vectorized transformation of FHA vectors to the global reference frame
    # einsum for batched matrix-vector multiplication
    transformed_hax = np.einsum('nij,nj->ni', R1_transform, hax) # Assuming hax are column vectors
    
    # Augment svec for homogeneous transformation
    svec_aug = np.hstack((svec, np.ones((svec.shape[0], 1))))
    transformed_svec_hom = np.einsum('nij,nj->ni', T1_transform, svec_aug)
    transformed_svec = transformed_svec_hom[:, :3]
    
    # Vectorized calculation of points 'p' on the axes
    p = transformed_svec + d[:, np.newaxis] * transformed_hax

    ##### AHA calculation (already vectorized with np.mean) #####
    axis_scale = 20
    transformed_average_hax = np.mean(transformed_hax, axis=0)
    transformed_average_hax /= np.linalg.norm(transformed_average_hax)
    transformed_average_svec = np.mean(transformed_svec, axis=0)
    average_d = np.mean(d)
    transformed_average_p = transformed_average_svec + average_d * transformed_average_hax
    transformed_average_start = transformed_average_p
    transformed_average_end = transformed_average_start + axis_scale * transformed_average_hax
    midpoint = (transformed_average_start + transformed_average_end) / 2

    # Define the plane size and resolution
    plane_size = 2  # Half the size of the square plane (adjust for visualization)
    plane_resolution = 10  # Number of points along each axis of the plane
    

    normal = transformed_average_hax

    arbitrary_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])

        # Compute the first orthogonal vector
    orthogonal1 = np.cross(normal, arbitrary_vector)
    orthogonal1 = orthogonal1 / np.linalg.norm(orthogonal1)  # Normalize
    
    # Compute the second orthogonal vector
    orthogonal2 = np.cross(normal, orthogonal1)
    orthogonal2 = orthogonal2 / np.linalg.norm(orthogonal2)  # Normalize
    


    u = np.linspace(-plane_size, plane_size, plane_resolution)
    v = np.linspace(-plane_size, plane_size, plane_resolution)
    uu, vv = np.meshgrid(u, v)

    # Map the grid onto the plane using the orthogonal basis vectors
    xx = midpoint[0] + uu * orthogonal1[0] + vv * orthogonal2[0]
    yy = midpoint[1] + uu * orthogonal1[1] + vv * orthogonal2[1]
    zz = midpoint[2] + uu * orthogonal1[2] + vv * orthogonal2[2]
    
    ########## Vectorized calculation of intersection points ###############
    intersection_points = calculate_intersection_vectorized(
        fha_origins=p,
        fha_directions=transformed_hax,
        plane_point=midpoint,
        plane_normal=normal
    )

    ########## Vectorized calculation of angles between FHAs and AHA ###############
    normalized_hax = transformed_hax / np.linalg.norm(transformed_hax, axis=1, keepdims=True)
    dot_products = np.clip(np.dot(normalized_hax, transformed_average_hax), -1.0, 1.0)
    angles_degrees = np.degrees(np.arccos(dot_products))
    average_angle = np.mean(angles_degrees)
    print(f"Average angle between FHA and AHA: {average_angle:.2f} degrees")
    
    # Reduce data for plotting
    p_small = p[::nn]
    transformed_hax_small = transformed_hax[::nn]
    translation_1_list_small = np.array(translation_1_list)[::nn]
    time_incr_small = np.array(time_incr)[::nn] if len(time_incr) > 0 else []
    ang_incr_small = np.array(ang_incr)[::nn] if len(ang_incr) > 0 else []

    # Setup Plot
    # fig = make_subplots(
    #     rows=4, cols=1, 
    #     specs=[[{"type": "scatter3d"}], [{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]],
    #     row_heights=[0.65, 0.05, 0.2, 0.2],
    #     vertical_spacing=0.08
    # )

    fig = make_subplots(
    rows=4, cols=1, 
    specs=[[{"type": "scatter3d"}], [{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]],
    row_heights=[0.65, 0.15, 0.1, 0.1],
    vertical_spacing=0.08
)

    # Plot FHA lines
    lines_x, lines_y, lines_z = [], [], []
    for i in range(len(p_small)):
        start = p_small[i]
        end = start + axis_scale * transformed_hax_small[i]
        lines_x.extend([start[0], end[0], None]) # Use None to break line segments
        lines_y.extend([start[1], end[1], None])
        lines_z.extend([start[2], end[2], None])
    
    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color='grey', width=2),
        name='FHA Axes'
    ), row=1, col=1)

    # Plot FHA origin points with color scale
    time_data = t if method_type == 'all_FHA' else time_incr
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2],
        mode='markers',
        marker=dict(
            color=time_data,
            colorscale='magma',
            size=4,
            colorbar=dict(title='Time (s)', orientation="h", y=0.1, x=0.5, len=0.8, thickness=15)
        ),
        name='FHA Origins'
    ), row=1, col=1)

    # Plot translation trajectories
    fig.add_trace(go.Scatter3d(
        x=np.array(translation_1_list)[:, 0], y=np.array(translation_1_list)[:, 1], z=np.array(translation_1_list)[:, 2],
        mode='lines+markers', marker=dict(size=2, color='black'), line=dict(color='black'), name='Translation 1'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter3d(
        x=np.array(translation_2_list)[:, 0], y=np.array(translation_2_list)[:, 1], z=np.array(translation_2_list)[:, 2],
        mode='lines+markers', marker=dict(size=2, color='green'), line=dict(color='green'), name='Translation 2'
    ), row=1, col=1)

    # Plot AHA
    fig.add_trace(go.Scatter3d(
        x=[transformed_average_start[0], transformed_average_end[0]], y=[transformed_average_start[1], transformed_average_end[1]], z=[transformed_average_start[2], transformed_average_end[2]],
        mode='lines', line=dict(color='blue', width=6), name='Average Helical Axis'
    ), row=1, col=1)

    # Plot the perpendicular plane
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        opacity=0.5,
        colorscale=[[0, 'cyan'], [1, 'cyan']],
        showscale=False,
        name='Perpendicular Plane'
    ), row=1, col=1)


    # Plot intersection points
    if intersection_points is not None and np.any(~np.isnan(intersection_points)):
        valid_intersections = intersection_points[~np.isnan(intersection_points).any(axis=1)]
        fig.add_trace(go.Scatter3d(
            x=valid_intersections[:, 0], y=valid_intersections[:, 1], z=valid_intersections[:, 2],
            mode='markers', 
            # marker=dict(size=4,     color='magenta'),
            marker=dict(
            color=time_data,
            colorscale='magma',
            size=4,
            # colorbar=dict(title='Time (s)', orientation="h", y=0.1, x=0.5, len=0.8, thickness=15)
        ),
                        

        name='Intersection Points'
        ), row=1, col=1)

    # --- 2D Plots ---
    plot_time = t[1:] if method_type == 'all_FHA' else time_incr
    plot_angle = all_angles if method_type == 'all_FHA' else ang_incr
    plot_mode = 'lines' if method_type in ['all_FHA', 'incremental_time'] else 'markers'

    fig.add_trace(go.Scatter(x=plot_time, y=plot_angle, mode=plot_mode, name='Angle'), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_time, y=time_diff, mode='lines', name='Time Difference'), row=4, col=1)

    if euler_angles.shape[0] > 0:
        fig.add_trace(go.Scatter(x=plot_time, y=euler_angles[:, 0], mode=plot_mode, name='Roll (X)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_time, y=euler_angles[:, 1], mode='lines', name='Pitch (Y)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_time, y=euler_angles[:, 2], mode='lines', name='Yaw (Z)'), row=2, col=1)

    
    # Update layout and axes
    fig.update_layout(
        title=f'FHA Visualization - {method_type}',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig.update_yaxes(title_text='Angle (°)', row=3, col=1)
    fig.update_xaxes(title_text='Time (s)', row=4, col=1)
    fig.update_yaxes(title_text='Time diff (s)', row=4, col=1)
    fig.update_yaxes(visible=False, showticklabels=False, row=2, col=1) # Hide dummy axis for colorbar
    fig.update_xaxes(visible=False, showticklabels=False, row=2, col=1)
    
    return fig