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
# import dash_core_components as dcc
from dash import dcc
    
def decompose_homogeneous_matrix(H):
    Horig = np.array(H)
    R = Horig[0:3,0:3]
    v = Horig[0:3,3].transpose()
    
    return R, v

def calculate_FHA(T1, T2):
    #Takes two homogeneous matrices as an input and returns parameters for the FHA associated with the rototranslation between them
    
    #Parameters returned:
        #n: normalized vector representing the direction of the FHA
        #phi: angle around the FHA
        #t: displacement along the FHA
        #s: location of the FHA (application point for the n vector)
    
    H = np.dot(T2, np.linalg.inv(T1))   #In this case, we post-multiply (INTRINSIC rotation: every subsequent rotation is
                                        #referred to the reference system of previous pose, and not to the global one)

    ROT, v = decompose_homogeneous_matrix(H)
    
    sinPhi = 0.5*np.sqrt(pow(ROT[2,1]-ROT[1,2],2)+pow(ROT[0,2]-ROT[2,0],2)+pow(ROT[1,0]-ROT[0,1],2))
    
    #CAREFUL: this calculation for cosine only works when sinPhi > sqrt(2)/2
    cosPhi=0.5*(np.trace(ROT)-1)
    
    #Implementing this condition, can use cosPhi calculated as before to estimate phi
    if sinPhi <= (np.sqrt(2)/2):
        # phi = math.degrees(np.arcsin(sinPhi))     #deg
        phi = np.arcsin(sinPhi)                     #rad
    else:
        # phi = math.degrees(np.arccos(cosPhi))     #deg
        phi = np.arccos(cosPhi)                     #rad
        
    n = (1/(2*sinPhi))*np.array([ROT[2,1]-ROT[1,2], ROT[0,2]-ROT[2,0], ROT[1,0]-ROT[0,1]])
    t = np.dot(n, np.array(v.transpose()))
    
    #The vector s (location of the FHA) should be calculated re-estimating sine and cosine of phi
    #through traditional functions (once phi is obtained), not using the sinPhi and cosPhi estimated from
    #the rotation matrix, because that calculation only works for sinPhi > sqrt(2)/2
    s = np.cross(-0.5*n, np.cross(n, v)) + np.cross((np.sin(phi)/(2*(1-np.cos(phi))))*n, v)
    
    return phi, n, t, s



def activate_method(method_type, t, Trel, loc1, loc2, all_hax, all_angles, all_svec, all_d, all_translation_1_list, all_translation_2_list, step):
    hax, ang, svec, d, translation_1_list, translation_2_list = [], [], [], [], [], []
    ind_incr, ind_step = [], []
    
    if(method_type != 'all_FHA'):    
        time_diff = []

    time_incr, ang_incr = [], []

    incrAng = []
    indCount = []

    if method_type == 'all_FHA':
        hax = all_hax
        ang = all_angles
        svec = all_svec
        d = all_d

        translation_1_list = all_translation_1_list
        translation_2_list = all_translation_2_list

    elif method_type == 'incremental_time':
        # Incremental method (time-based)
        incrAng = []
        angSum = 0
        ind_step = [0]
        for i in range(len(all_angles)):
            angSum += all_angles[i]
            incrAng.append(angSum)
        
        for i in range(len(t) - step):
            phi, n, tran, s = calculate_FHA(Trel[i], Trel[i + step])
            hax.append(n)
            ang.append(phi)
            svec.append(s)
            d.append(tran)
            
            translation_1_list.append(loc1[i])
            translation_2_list.append(loc2[i])
            
            time_incr.append(t[i])
            ang_incr.append(incrAng[i])
            time_diff.append(t[i+step]-t[i])
    
    elif method_type == 'step_angle':
        # Incremental method (step-based)
        incrAng = []
        angSum = 0
        ind_step = [0]
        for i in range(len(all_angles)):
            angSum += all_angles[i]
            incrAng.append(angSum)
            if angSum > step:
                ind_step.append(i)
                angSum = 0  # Reset only after adding an index

        for i in range(1, len(ind_step)):
            phi, n, tran, s = calculate_FHA(Trel[ind_step[i-1]], Trel[ind_step[i]])
            hax.append(n)
            ang.append(phi)
            svec.append(s)
            d.append(tran)

            translation_1_list.append(loc1[ind_step[i-1]])
            translation_2_list.append(loc2[ind_step[i-1]])
            
            time_incr.append(t[ind_step[i-1]])
            ang_incr.append(incrAng[ind_step[i]])           
            time_diff.append(t[ind_step[i]]-t[ind_step[i-1]])

    elif method_type == 'incremental_angle':
        # Incremental method (angle-based)
        incrAng = []
        angSum = 0
        ind_incr = []
        for i in range(len(all_angles) - 1):
            angSum = 0
            for j in range(i, len(all_angles)):
                angSum += all_angles[j]
                incrAng.append(angSum)
                indCount.append([i,j])
                if angSum > step:
                    ind_incr.append([i, j])
                    break
        
        # ang_incr_ang.append(incrAng[ind_incr[0][1]])
        for i in range(0, len(ind_incr)):
            phi, n, tran, s = calculate_FHA(Trel[ind_incr[i][0]], Trel[ind_incr[i][1]])
            hax.append(n)
            ang.append(phi)
            svec.append(s)
            d.append(tran)

            translation_1_list.append(loc1[ind_incr[i][0]])
            translation_2_list.append(loc2[ind_incr[i][0]])
            time_incr.append(t[ind_incr[i][0]])
            indSum = [j for j in range(len(indCount)) if indCount[j] == ind_incr[i]]
            ang_incr.append(incrAng[indSum[0]])
            time_diff.append(t[ind_incr[i][1]]-t[ind_incr[i][0]])

    else:
        raise ValueError("Invalid method type. Choose from 'all_FHA', 'incremental_time', 'step_angle', or 'incremental_angle'.")
    
    
    return hax, ang, svec, d, translation_1_list, translation_2_list, ind_incr, ind_step, incrAng, time_incr, ang_incr


def generate_FHA(method_type, t, cut1, cut2, step, nn, R1, R2, T1, T2, loc1, loc2):

    ###### Cut data ######
    R1=R1[cut1:cut2]
    R2=R2[cut1:cut2]
    T1=T1[cut1:cut2]
    T2=T2[cut1:cut2]
    loc1=loc1[cut1:cut2]
    loc2=loc2[cut1:cut2]
    t=t[cut1:cut2]

    ##### Define value of steps and the number of samples to skip for plotting #####
    if method_type == 'all_FHA':
        step = None # Not used
        nn = nn
    elif method_type == 'incremental_time':
        step = step # amount of samples to skip
        nn = nn
    elif method_type == 'step_angle':
        step = step # amount of degrees to skip
        nn = nn
    elif method_type == 'incremental_angle':
        step = step # amount of degrees to increment
        nn = nn


    T1inv=[]
    for i in range(len(T1)):
        T1inv.append(np.linalg.inv(T1[i]))
    
    T1inv = np.array(T1inv)
        
    Trel = []
    for i in range(len(T1inv)):
        # Note: we pre-multiply Tinv, meaning an EXTRINSIC rotation 
        # (result of multiple combined rotations is always referred to the orignal reference system) 
        Trel.append(np.dot(T1inv[i,:,:], T2[i,:,:]))

    Trel=np.array(Trel)
    
    all_hax = []
    all_angles = []
    all_svec = []
    all_d = []

    all_translation_1_list = []
    all_translation_2_list = []

    time_diff = []

    # Calculate all helical axis:
    for i in range(len(Trel)-1):
        phi, n, d, s = calculate_FHA(Trel[i], Trel[i+1])
        all_hax.append(n)
        # ang.append(phi)
        # all_angles.append(phi)
        all_angles.append(math.degrees(phi))


        all_svec.append(s)
        all_d.append(d)
        
        all_translation_1_list.append(loc1[i])
        all_translation_2_list.append(loc2[i])
        
        time_diff.append(t[i+1]-t[i])
        
    hax, ang, svec, d, translation_1_list, translation_2_list, ind_incr, ind_step, incrAng, time_incr, ang_incr = activate_method(method_type, t, Trel, loc1, loc2, all_hax, all_angles, all_svec, all_d, all_translation_1_list, all_translation_2_list, step)

    return hax, ang, svec, d, translation_1_list, translation_2_list, time_diff, time_incr, ang_incr, all_angles, ind_incr, ind_step, t


def calculate_intersection(fha_origin, fha_direction, plane_point, plane_normal):
    """
    Calculate the intersection of a line with a plane.

    Parameters:
    - fha_origin: Origin point of the finite helical axis (3D array).
    - fha_direction: Direction vector of the finite helical axis (3D array, normalized).
    - plane_point: A point on the plane (3D array).
    - plane_normal: The normal vector of the plane (3D array, normalized).

    Returns:
    - Intersection point (3D array).
    """
    # Ensure the direction and normal vectors are normalized
    fha_direction = fha_direction / np.linalg.norm(fha_direction)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Calculate t (intersection parameter)
    numerator = np.dot(plane_point - fha_origin, plane_normal)
    denominator = np.dot(fha_direction, plane_normal)
    if np.isclose(denominator, 0):  # Check for parallelism
        return None  # No intersection

    t = numerator / denominator

    # Calculate intersection point
    intersection = fha_origin + t * fha_direction
    return intersection

def plotly_visualize_FHA(isPolhemus, T1, T2, method_type, hax, svec, d, ind_incr, ind_step, nn, translation_1_list, translation_2_list, t, time_incr, ang_incr, step, all_angles, time_diff):

    #transform into sensor1 reference system for plotting
    R1=[]
    for i in range(len(T1)):
        ROT,v = decompose_homogeneous_matrix(T1[i])
        R1.append(ROT)
    R1=np.array(R1)
    
    transformed_hax = []
    transformed_svec = []
    p = []
    
    if method_type == 'all_FHA':
        # all_FHA
        for i in range(len(hax)):
            transformed_hax.append(np.dot(hax[i], R1[i]))
            transformed_svec.append(np.dot(T1[i], np.append(svec[i], 1).transpose()))
            p.append(transformed_svec[i][0:3] + d[i]*transformed_hax[i])
    elif method_type == 'incremental_time':
        # incremental_time
        for i in range(len(hax)):
            transformed_hax.append(np.dot(hax[i], R1[i]))
            transformed_svec.append(np.dot(T1[i], np.append(svec[i], 1).transpose()))
            p.append(transformed_svec[i][0:3] + d[i]*transformed_hax[i])
    elif method_type == 'step_angle':
        # incremental_step
        for i in range(len(hax)):
            transformed_hax.append(np.dot(hax[i], R1[ind_step[i]]))
            transformed_svec.append(np.dot(T1[ind_step[i]], np.append(svec[i], 1).transpose()))
            p.append(transformed_svec[i][0:3] + d[i]*transformed_hax[i])
    elif method_type == 'incremental_angle':
        # incremental_angle
        for i in range(len(hax)):
            transformed_hax.append(np.dot(hax[i], R1[ind_incr[i][0]]))
            transformed_svec.append(np.dot(T1[ind_incr[i][0]], np.append(svec[i], 1).transpose()))
            p.append(transformed_svec[i][0:3] + d[i]*transformed_hax[i])
    
    ##### AHA calculation #####
    
    axis_scale = 20  # Adjust scale for visualization
    
    transformed_average_hax = np.mean(transformed_hax, axis=0)
    transformed_average_hax = transformed_average_hax / np.linalg.norm(transformed_average_hax)  # Normalize
    
    transformed_average_svec = np.mean(transformed_svec, axis=0)
    transformed_average_svec = transformed_average_svec[:3]
    
    average_d = np.mean(d, axis=0)
    
    transformed_average_p = transformed_average_svec + average_d * transformed_average_hax
    
    # Extend the AHA line for plotting
    transformed_average_start = transformed_average_p
    transformed_average_end = transformed_average_start + axis_scale * transformed_average_hax
    
    # Calculate the midpoint of the AHA
    midpoint = (transformed_average_start + transformed_average_end) / 2
    
    # Define the plane size and resolution
    plane_size = 10  # Half the size of the square plane (adjust for visualization)
    plane_resolution = 10  # Number of points along each axis of the plane
    
    # Normal vector is the AHA direction
    normal = transformed_average_hax/np.linalg.norm(transformed_average_hax)
    
    # Create two orthogonal vectors to the normal to define the plane's basis
    # Start with any vector not parallel to the normal
    arbitrary_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    
    # Compute the first orthogonal vector
    orthogonal1 = np.cross(normal, arbitrary_vector)
    orthogonal1 = orthogonal1 / np.linalg.norm(orthogonal1)  # Normalize
    
    # Compute the second orthogonal vector
    orthogonal2 = np.cross(normal, orthogonal1)
    orthogonal2 = orthogonal2 / np.linalg.norm(orthogonal2)  # Normalize
    
    # Create a grid of points in the local plane coordinate system
    u = np.linspace(-plane_size, plane_size, plane_resolution)
    v = np.linspace(-plane_size, plane_size, plane_resolution)
    uu, vv = np.meshgrid(u, v)
    
    # Map the grid onto the plane using the orthogonal basis vectors
    xx = midpoint[0] + uu * orthogonal1[0] + vv * orthogonal2[0]
    yy = midpoint[1] + uu * orthogonal1[1] + vv * orthogonal2[1]
    zz = midpoint[2] + uu * orthogonal1[2] + vv * orthogonal2[2]
    
    ########## Calculate angles between each FHA and the average helical axis ###############
    
    # Normalize each helical axis
    normalized_hax = transformed_hax / np.linalg.norm(transformed_hax, axis=1)[:, None]
    
    # Compute angles with the average helical axis
    dot_products = np.dot(normalized_hax, transformed_average_hax)  # Dot product with AHA
    # dot_products = np.clip(dot_products, -1.0, 1.0)  # Clip to avoid numerical errors outside valid range
    angles_radians = np.arccos(dot_products)  # Angle in radians
    angles_degrees = np.degrees(angles_radians)  # Convert to degrees
    
    # Print or use the calculated angles
    # print("Angles (degrees):", angles_degrees)
    
    # calculate the average angle between the FHA and the AHA
    average_angle = np.mean(angles_degrees)
    print("Average angle (degrees):", average_angle)
    
    
    # Calculate intersections
    intersection_points = []
    
    for i in range(len(hax)):
        intersection = calculate_intersection(
            fha_origin=p[i],
            fha_direction=transformed_hax[i],
            plane_point=midpoint,
            plane_normal=normal
        )
        if intersection is not None:
            intersection_points.append(intersection)
    
    # Convert to array for easier manipulation
    intersection_points = np.array(intersection_points)
    
    
    #reduce data dimensionality for plotting (1 sample every nn)
    hax_small = hax[::nn]
    svec_small = svec[::nn]
    d_small = d[::nn]
    
    p_small = p[::nn]
    
    transformed_hax_small = transformed_hax[::nn]
    transformed_svec_small = transformed_svec[::nn]
    
    translation_1_list_small = translation_1_list[::nn]
    translation_2_list_small = translation_2_list[::nn]
    
    intersections_points_small = intersection_points[::nn]
    
    t_small = t[::nn] 
    time_incr_small = time_incr[::nn]
    ang_incr_small = ang_incr[::nn]    

    # Create 3D scatter and line traces
    time_data = np.linspace(0,1,len(t)) if method_type == 'all_FHA' else np.linspace(0,1,len(time_incr))
    time_data_small = np.linspace(0,1,len(t_small)) if method_type == 'all_FHA' else np.linspace(0,1,len(time_incr_small))
    
    
    # color_norm = (time_data - np.min(time_data)) / (np.max(time_data) - np.min(time_data))
    # colors = [f'rgba({int(c*255)}, {int(100)}, {int(0)}, 1)' for c in color_norm]
    
    # color_norm_small = (time_data_small - np.min(time_data_small)) / (np.max(time_data_small) - np.min(time_data_small))
    # colors_small = [f'rgba({int(c*255)}, {int(100)}, {int(0)}, 1)' for c in color_norm_small]
    
    n_colors = len(time_data)
    colors = px.colors.sample_colorscale("magma", [n/(n_colors -1) for n in range(n_colors)])
    
    n_colors = len(time_data_small)
    colors_small = px.colors.sample_colorscale("magma", [n/(n_colors -1) for n in range(n_colors)])

    
    # fig = go.Figure()
    r=4
    c=1
    fig = make_subplots(rows=r, cols=1, specs=[[{"type": "scatter3d"}], 
        [{"type": "scatter"}],
        [{"type": "scatter"}],
        [{"type": "scatter"}]],
        row_heights = [0.65,0.05,0.2,0.2])

    # Plot each FHA line
    for i in range(len(hax_small)):
        start = p_small[i]
        end = start + axis_scale * transformed_hax_small[i]

        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color=colors_small[i], width=3),
            # name='FHA Axis'
        ), row=1, col=1)
        
    pX=[]
    pY=[]
    pZ=[]
        
    for i in range(len(hax_small)):
        start = p_small[i]
        end = p_small[i] + transformed_hax_small[i]*axis_scale
        

        pX.append(p_small[i][0])
        pY.append(p_small[i][1])
        pZ.append(p_small[i][2])
    
    # Colored scatter of FHA origin points
    fig.add_trace(
        go.Scatter3d(
            x=pX, y=pY, z=pZ,
            mode='markers',
            marker=dict(
                color=time_data,
                colorscale='magma',
                size=4,
                # colorbar=dict(
                #     # title='Time (s)',
                #     len=1,
                #     thickness=15,
                #     orientation = "h",
                #     y=0.65,
                # )
            ),
            name='FHA Origins'
        ), row=1, col=1
    )
    
    colorbar_trace=go.Scatter(
            # x=[None],
            # y=[None],
            x=[0,1],
            y= [2,2],
            fill='tozeroy',
            fillcolor  = 'white',
            mode='markers',
            marker=dict(
                color=time_data,
                colorscale='magma',
                size=4,
                colorbar=dict(
                    # title='Time (s)',
                    len=1,
                    thickness=15,
                    orientation = "h",
                    y=0.4,
                )
            ),
            hoverinfo='none'
    )

    layout = dict(xaxis=dict(visible=False), yaxis=dict(visible=False))



    # Plot translation points
    trans_x = []
    trans_y = []
    trans_z = []
    for i in range(len(hax_small)):
        trans_x.append(translation_1_list_small[i][0])
        trans_y.append(translation_1_list_small[i][1])
        trans_z.append(translation_1_list_small[i][2])
        
    fig.add_trace(go.Scatter3d(
        x=trans_x,
        y=trans_y,
        z=trans_z,
        mode='markers',
        marker=dict(size=4, color='black'),
        name='Translation 1'
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in translation_2_list],
        y=[p[1] for p in translation_2_list],
        z=[p[2] for p in translation_2_list],
        mode='markers',
        marker=dict(size=4, color='green'),
        name='Translation 2'
    ), row=1, col=1)

    # Plot average FHA line
    fig.add_trace(go.Scatter3d(
        x=[transformed_average_start[0], transformed_average_end[0]],
        y=[transformed_average_start[1], transformed_average_end[1]],
        z=[transformed_average_start[2], transformed_average_end[2]],
        mode='lines',
        line=dict(color='blue', width=6),
        name='Average Helical Axis'
    ), row=1, col=1)

    # Plot the perpendicular plane
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        opacity=0.5,
        colorscale=[[0, 'cyan'], [1, 'cyan']],
        showscale=False,
        name='Perpendicular Plane'
    ), row=1, col=1)


    # Plot midpoint
    fig.add_trace(go.Scatter3d(
        x=[midpoint[0]],
        y=[midpoint[1]],
        z=[midpoint[2]],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Midpoint'
    ), row=1, col=1)

    # Plot intersection points if any
    if intersection_points is not None and len(intersection_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=intersection_points[:, 0],
            y=intersection_points[:, 1],
            z=intersection_points[:, 2],
            mode='markers',
            marker=dict(size=4, color='magenta'),
            name='Intersection Points'
        ), row=1, col=1)
        
    fig.add_trace(colorbar_trace, row=2, col=1)
    # fig.update_layout(layout)
        
        
    if method_type == 'all_FHA':
        fig.add_trace(
            go.Scatter(
                x=t[1:], y=all_angles,
                mode='lines',
                name='Angle (All FHA)'
            ), row=3, col=1
        )
    elif method_type == 'incremental_time':
        fig.add_trace(
            go.Scatter(
                x=time_incr, y=ang_incr,
                mode='lines',
                name='Angle (Incremental Time)'
            ), row=3, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=time_incr, y=ang_incr,
                mode='markers',
                name='Angle Points'
            ), row=3, col=1
        )

        # Optional: Set y-range based on `step` for this method
        fig.update_yaxes(
            range=[step - (step / 100) * 20, step + (step / 100) * 20],
            row=3, col=1
        )

    # --- Second Plot: Time Difference ---
    fig.add_trace(
        go.Scatter(
            x=np.linspace(1, t[-1], len(time_diff)),
            y=time_diff,
            mode='lines',
            name='Time Difference'
        ), row=4, col=1
    )

    # Update axis labels
    fig.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig.update_yaxes(title_text='Angle (°)', row=3, col=1)
    fig.update_xaxes(title_text='Time (s)', row=4, col=1)
    fig.update_yaxes(title_text='Time diff', row=4, col=1)

    fig.update_layout(
        title=f'FHA Visualization - {method_type}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=False,
        # legend=dict(x=0.1, y=0.9),
    )

    # fig.show()
    
    return fig