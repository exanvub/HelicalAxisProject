# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 13:53:48 2025

@author: Matteo Iurato
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R
import math
import time

plt.close('all')
    
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


def visualize_FHA(isPolhemus, T1, T2, method_type, hax, svec, d, ind_incr, ind_step, nn, translation_1_list, translation_2_list, t, time_incr, ang_incr, step, all_angles, time_diff):
    
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
    
    
    #plot
    fig = plt.figure()
    axis_scale = 20  
    ax1 = fig.add_axes([0.25, 0.5, 0.50, 0.50], projection='3d')
    
    pX=[]
    pY=[]
    pZ=[]
    
    cmap = mpl.colormaps['autumn_r']
    if(method_type == 'all_FHA'):
        colors = cmap(np.linspace(0,1,len(t)))
    else:
        colors = cmap(np.linspace(0,1,len(time_incr)))
        
    if(method_type == 'all_FHA'):
        colors_small = cmap(np.linspace(0,1,len(t_small)))
    else:
        colors_small = cmap(np.linspace(0,1,len(time_incr_small)))
        
    
    for i in range(len(hax_small)):
        start = p_small[i]
        end = p_small[i] + transformed_hax_small[i]*axis_scale
        
        ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=colors_small[i])
        pX.append(p_small[i][0])
        pY.append(p_small[i][1])
        pZ.append(p_small[i][2])
        
        ax1.scatter(translation_1_list_small[i][0], translation_1_list_small[i][1], translation_1_list_small[i][2], color='k')
        ax1.scatter(translation_2_list_small[i][0], translation_2_list_small[i][1], translation_2_list_small[i][2], color='g')
        
    ax1.scatter(translation_2_list_small[0][0], translation_2_list_small[0][1], translation_2_list_small[0][2], color='k', s=10)     
    # ax1.scatter(marker_data['caput']['x'], marker_data['caput']['y'], marker_data['caput']['z'], color='k', s=10) #pre-name variables
    
    if(method_type == 'all_FHA'):
        sc=ax1.scatter(pX, pY, pZ, c=t_small, cmap='autumn_r')
    else:
        sc=ax1.scatter(pX, pY, pZ, c=time_incr_small, cmap='autumn_r')
        
    if(isPolhemus == True):
        #this is to align the plotting reference frame with the Polhemus transmitter reference frame (needed if data were acquired using the default reference system)
        ax1.view_init(elev=180) 
    
    # Add the transformed average helical axis to the plot
    ax1.plot([transformed_average_start[0], transformed_average_end[0]],
            [transformed_average_start[1], transformed_average_end[1]],
            [transformed_average_start[2], transformed_average_end[2]], 'b-', linewidth=2, label='Average Helical Axis')
    
    ax1.scatter(transformed_average_start[0], transformed_average_start[1], transformed_average_start[2], color='b', s=10, label='AHA Position')
    
    ax1.scatter(midpoint[0], midpoint[1], midpoint[2], color='r', s=10, label='Perpendicular Plane Midpoint')
    
    # ax1.plot_surface(xx, yy, zz, alpha=0.5, color='cyan', edgecolor='none', label='Perpendicular Plane')
    
    # Plot intersection points
    # ax1.scatter(intersections_points_small[:, 0], intersections_points_small[:, 1], intersections_points_small[:, 2], color='magenta', label='Intersections')
    
    # Equal axis scaling
    x_limits = ax1.get_xlim3d()
    y_limits = ax1.get_ylim3d()
    z_limits = ax1.get_zlim3d()
    
    max_range = np.array([x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0]]).max() / 4.0
    
    mid_x = (x_limits[1] + x_limits[0]) * 0.5
    mid_y = (y_limits[1] + y_limits[0]) * 0.5
    mid_z = (z_limits[1] + z_limits[0]) * 0.5
    
    ax1.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax1.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax1.set_zlim3d([mid_z - max_range, mid_z + max_range])    
    
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    
    ax2 = fig.add_axes([0.15, 0.45, 0.7, 0.03])  
    ax2.set_ylim([0,0.5])
    cbar=fig.colorbar(sc, cax=ax2, orientation='horizontal', pad=0.05)
    cbar.set_label('Time (s)')
    
    ax3 = fig.add_axes([0.15, 0.3, 0.7, 0.1])  
    if(method_type == 'all_FHA'):
        ax3.plot(t[1:], all_angles)   
    elif(method_type == 'incremental_time'):
        ax3.plot(time_incr, ang_incr)
    else:
        ax3.plot(time_incr, ang_incr, 'o')
        ax3.set_ylim([step-(step/100)*20, step+(step/100)*20]) 
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (°)')
    
    ax4 = fig.add_axes([0.15, 0.1, 0.7, 0.1])
    ax4.plot(np.linspace(1,t[-1],len(time_diff)), time_diff)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Time diff')
    
    fig.suptitle(method_type)
    plt.show(block=False)
    
    # Transform the points to the plane's local coordinate system
    plane_points_2d = []
    
    for point in p:
        # Vector from the midpoint of the plane to the point
        point_vector = point - midpoint
        
        # Project onto the plane's orthogonal basis vectors
        u_coord = np.dot(point_vector, orthogonal1)
        v_coord = np.dot(point_vector, orthogonal2)
        
        plane_points_2d.append([u_coord, v_coord])
    
    plane_points_2d = np.array(plane_points_2d)
    
    # Plot the points in 2D
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    # plt.scatter(plane_points_2d[:, 0], plane_points_2d[:, 1], c='b', label='FHA Intersection Points')
    for i in range(len(plane_points_2d)):
        ax.scatter(plane_points_2d[i, 0], plane_points_2d[i, 1], color=colors[i])
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
    plt.grid(True, linestyle=':')
    plt.xlabel('U (Plane Axis 1)')
    plt.ylabel('V (Plane Axis 2)')
    plt.title('FHA Intersection Points in Plane Coordinate System')
    plt.legend()
    plt.show()
    
