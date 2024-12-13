# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:21:23 2024

@author: Matteo Iurato
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R
import math

plt.close('all')


#%% Data reading

data = pd.read_csv('../test_data/Polhemus_test_data/1_90deg_ydata.csv')


# Extract relevant data
time = data['Time'].values
q1 = data[['w1', 'x1', 'y1', 'z1']].values
q2 = data[['w2', 'x2', 'y2', 'z2']].values
loc1 = data[['loc1_x', 'loc1_y', 'loc1_z']].values
loc2 = data[['loc2_x', 'loc2_y', 'loc2_z']].values


# Use following lines if you want to cut the beginning and end of the acquisition
# This might be necessary because stationary conditions (no movement) introduce high error in the FHA position calculation
# due to very small difference between one timeframe and the subsequent

# cut1=500
# cut2=1900
# q1=q1[cut1:cut2]
# q2=q2[cut1:cut2]
# loc1=loc1[cut1:cut2]
# loc2=loc2[cut1:cut2]
# time=time[cut1:cut2]

#%% Quaternion to rotation matrix conversion

# Manual implementation of quaternion to rotation matrix conversion
# def quaternion_to_rotation_matrix(w, x, y, z):
#     r11 = pow(w, 2) + pow(x, 2) - pow(y, 2) - pow(z, 2)
#     r12 = 2*x*y - 2*w*z
#     r13 = 2*x*y+2*w*y
#     r21 = 2*x*y+2*w*z
#     r22 = pow(w, 2) - pow(x, 2) + pow(y, 2) - pow(z, 2)
#     r23 = 2*y*z - 2*w*x
#     r31 = 2*x*z - 2*w*y
#     r32 = 2*y*z + 2*w*z
#     r33 = pow(w, 2) - pow(x, 2) - pow(y, 2) + pow(z, 2)
    
#     ROT = np.matrix([[r11, r12, r13],
#                     [r21, r22, r23],
#                     [r31, r32, r33]])
    
#     return ROT

# Scipy quaternion to rotation matrix convertion
def quaternion_to_rotation_matrix(quaternion):
    quat = np.roll(quaternion, -1) #rearrange quaternion components as x,y,z,w for scipy
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

R1 = []
R2 = []
for i in range(len(q1)):
    
    # Use this if you rely on manual conversion
    # R1.append(quaternion_to_rotation_matrix(q1[i][0], q1[i][1], q1[i][2], q1[i][3]))
    # R2.append(quaternion_to_rotation_matrix(q2[i][0], q2[i][1], q2[i][2], q2[i][3]))
    
    # Use this if you rely on scipy conversion
    R1.append(quaternion_to_rotation_matrix(q1[i]))
    R2.append(quaternion_to_rotation_matrix(q2[i]))


#%% Homogeneous matrices building

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
    
#%% Calculate relative homogenous matrix

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


#%% Calculate FHA    
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
    

hax = []
ang = []
svec = []
d = []

translation_1_list = []
translation_2_list = []

# Traditional method
for i in range(len(Trel)-1):
    phi, n, t, s = calculate_FHA(Trel[i], Trel[i+1])
    hax.append(n)
    # ang.append(phi)
    ang.append(math.degrees(phi))
    svec.append(s)
    d.append(t)
    
    translation_1_list.append(loc1[i])
    translation_2_list.append(loc2[i])

# Incremental method (time-based)
# incr=5
# for i in range(len(time)-incr):
#     phi, n, t, s = calculate_FHA(Trel[i], Trel[i+incr])
#     hax.append(n)
#     # ang.append(phi)
#     ang.append(math.degrees(phi))
#     svec.append(s)
#     d.append(t)
    
#     translation_1_list.append(loc1[i])
#     translation_2_list.append(loc2[i])

# Angles step
step = 5
angSum = 0
ind_step = []
ind_step.append(0)
for i in range(len(ang)):
    angSum = 0
    angSum += ang[i]
    if(angSum > step):
        ind_step.append(i)

hax_step = []
ang_step = []
svec_step = []
d_step = []
for i in range(1, len(ind_step)):
    phi, n, t, s = calculate_FHA(Trel[ind_step[i-1]], Trel[ind_step[i]])
    hax_step.append(n)
    # ang_step.append(phi)
    ang_step.append(math.degrees(phi))
    svec_step.append(s)
    d_step.append(t)

# Incremental method (angle-based)
step = 10
angSum = 0
ind_incr = []
for i in range(len(ang)-1):
    angSum = 0
    for j in range(i, len(ang)):
        angSum += ang[j]
        if (angSum > step):
            ind_incr.append([i, j])
            break
        
        
        
hax_incr = []
ang_incr = []
svec_incr = []
d_incr = []
for i in range(1, len(ind_incr)):
    phi, n, t, s = calculate_FHA(Trel[ind_incr[i][0]], Trel[ind_incr[i][1]])
    hax_incr.append(n)
    # ang_incr.append(phi)
    ang_incr.append(math.degrees(phi))
    svec_incr.append(s)
    d_incr.append(t)

    
#%% Plotting 
nn = 25

def plot_fha(nn, fig, ind, T1, hax, svec, d, t1, t2, reduce):
    #transform into sensor1 reference system for plotting
    R1=[]
    for i in range(len(ind)):
        ROT,v = decompose_homogeneous_matrix(T1[ind[i]])
        R1.append(ROT)
    R1=np.array(R1)
    
    transformed_hax = []
    transformed_svec = []
    for i in range(len(hax)):
        transformed_hax.append(np.dot(hax[i], R1[i]))
        transformed_svec.append(np.dot(T1[i], np.append(svec[i], 1).transpose()))
    
    if (reduce == True):
        #reduce data dimensionality for plotting (1 sample every nn)
        hax_small = hax[::nn]
        svec_small = svec[::nn]
        d_small = d[::nn]
        
        transformed_hax_small = transformed_hax[::nn]
        transformed_svec_small = transformed_svec[::nn]
        
    else:
        hax_small = hax
        svec_small = svec
        d_small = d
        
        transformed_hax_small = transformed_hax
        transformed_svec_small = transformed_svec
        
    translation_1_list_small = translation_1_list[::nn]
    translation_2_list_small = translation_2_list[::nn]
        
    
    #plot
    # fig = plt.figure()
    axis_scale = 20  
    ax = fig.add_subplot(111, projection='3d') 
    
    
    for i in range(len(hax_small)):
        p = transformed_svec_small[i][0:3] + d_small[i]*transformed_hax_small[i]
        
        start = p 
        end = p + transformed_hax_small[i]*axis_scale
        
        pl = ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-')
        sc1 = ax.scatter(p[0], p[1], p[2], color='b', s=5)    
        
    for i in range(len(translation_1_list_small)):    
        sc2 = ax.scatter(translation_1_list_small[i][0], translation_1_list_small[i][1], translation_1_list_small[i][2], color='k')
        sc3 = ax.scatter(translation_2_list_small[i][0], translation_2_list_small[i][1], translation_2_list_small[i][2], color='g')
        
    ax.scatter(translation_2_list_small[0][0], translation_2_list_small[0][1], translation_2_list_small[0][2], color='k', s=50) 
    
    #this is to align the plotting reference frame with the Polhemus transmitter reference frame (needed if data were acquired using the default reference system)
    ax.view_init(elev=180) 
       
    
    # Equal axis scaling
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    max_range = np.array([x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0]]).max() / 4.0
    
    mid_x = (x_limits[1] + x_limits[0]) * 0.5
    mid_y = (y_limits[1] + y_limits[0]) * 0.5
    mid_z = (z_limits[1] + z_limits[0]) * 0.5
    
    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])    
    
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.legend(pl, 'FHA')
    ax.legend([pl, sc1, sc2, sc3], ['FHA', 'FHA Position', 'Sensor 1', 'Sensor 2'])

fig1 = plt.figure()
plot_fha(nn, fig1, np.linspace(0,len(T1)-1,len(T1)).astype(int), T1, hax, svec, d, translation_1_list, translation_2_list, True)

fig2 = plt.figure()
plot_fha(nn, fig2, ind_step, T1, hax_step, svec_step, d_step, translation_1_list, translation_2_list, False)

fig3 = plt.figure()
plot_fha(nn, fig3, [item[0] for item in ind_incr], T1, hax_incr, svec_incr, d_incr, translation_1_list, translation_2_list, True)

plt.show()