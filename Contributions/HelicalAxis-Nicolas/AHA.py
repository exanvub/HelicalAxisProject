import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R

plt.close('all')

data = pd.read_csv('test_data/Polhemus_test_data/1_90deg_ydata_oneway.csv')

# Extract relevant data
time = data['Time'].values
q1 = data[['w1', 'x1', 'y1', 'z1']].values
q2 = data[['w2', 'x2', 'y2', 'z2']].values
loc1 = data[['loc1_x', 'loc1_y', 'loc1_z']].values
loc2 = data[['loc2_x', 'loc2_y', 'loc2_z']].values

# cut1=1500
# cut2=2800
# q1=q1[cut1:cut2]
# q2=q2[cut1:cut2]
# loc1=loc1[cut1:cut2]
# loc2=loc2[cut1:cut2]
# time=time[cut1:cut2]

# Scipy quaternion to rotation matrix convertion
def quaternion_to_rotation_matrix(quaternion):
    quat = np.roll(quaternion, -1) #rearrange quaternion components as x,y,z,w for scipy
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

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

# # Traditional method
# for i in range(len(Trel)-1):
#     phi, n, t, s = calculate_FHA(Trel[i], Trel[i+1])
#     hax.append(n)
#     ang.append(phi)
#     svec.append(s)
#     d.append(t)
    
#     translation_1_list.append(loc1[i])
#     translation_2_list.append(loc2[i])

# Incremental method (time-based for now)
incr=20
for i in range(len(time)-incr):
    phi, n, t, s = calculate_FHA(Trel[i], Trel[i+incr])
    hax.append(n)
    ang.append(phi)
    svec.append(s)
    d.append(t)
    
    translation_1_list.append(loc1[i])
    translation_2_list.append(loc2[i])


#transform into sensor1 reference system for plotting
R1=[]
for i in range(len(T1)):
    ROT,v = decompose_homogeneous_matrix(T1[i])
    R1.append(ROT)
R1=np.array(R1)

transformed_hax = []
transformed_svec = []

for i in range(len(hax)):
    transformed_hax.append(np.dot(hax[i], R1[i]))
    transformed_svec.append(np.dot(T1[i], np.append(svec[i], 1).transpose()))

p = []
for i in range(len(hax)):
    p.append(transformed_svec[i][0:3] + d[i]*transformed_hax[i])

#reduce data dimensionality for plotting (1 sample every nn)
nn = 50

hax_small = hax[::nn]
svec_small = svec[::nn]
d_small = d[::nn]

p_small = p[::nn]

transformed_hax_small = transformed_hax[::nn]
transformed_svec_small = transformed_svec[::nn]

translation_1_list_small = translation_1_list[::nn]
translation_2_list_small = translation_2_list[::nn]

##### AHA calculation #####

# average direction vector (mean of unit vectors)
hax_array = np.array(hax)
average_hax = np.mean(hax_array, axis=0)
average_hax = average_hax / np.linalg.norm(average_hax)  # Normalize

# average position of the FHA points
svec_array = np.array(svec)
average_position = np.mean(svec_array, axis=0)

d_array = np.array(d)
average_d = np.mean(d_array, axis=0)

average_p = average_position + average_d * average_hax

# Extend the AHA line for plotting
axis_scale = 20  # Adjust scale for visualization
average_start = average_p
average_end = average_start + axis_scale * average_hax

# Transform average_position into Sensor 1's reference system
average_p_homogeneous = np.append(average_p, 1)  # Make it homogeneous (x, y, z, 1)

# Apply the transformation using T1[0] (or the initial transformation of Sensor 1)
transformed_average_p = np.dot(T1[0], average_p_homogeneous)

# Extract the transformed (x, y, z) coordinates
transformed_average_p = transformed_average_p[:3]

# Adjust the start and end of the AHA for plotting in Sensor 1's reference system
transformed_average_start = transformed_average_p
transformed_average_end = transformed_average_start + axis_scale * average_hax


# Calculate the midpoint of the AHA
midpoint = (transformed_average_start + transformed_average_end) / 2

# # Define the dimensions of the rectangular plane
# plane_width = 5   # Width of the plane along one axis
# plane_height = 5  # Height of the plane along the other axis
# plane_resolution = 10  # Number of points along each axis of the plane

# # Create a grid of points for the rectangular plane
# x_points = np.linspace(-plane_width / 2, plane_width / 2, plane_resolution)
# y_points = np.linspace(-plane_height / 2, plane_height / 2, plane_resolution)
# xx, yy = np.meshgrid(x_points, y_points)
# # Normal vector is the AHA direction
# normal = average_hax
# normal = normal / np.linalg.norm(normal)  # Ensure it's normalized

# # Use the plane equation: n · (x - p) = 0
# # Solve for z to get points on the plane
# zz = (-normal[0] * xx - normal[1] * yy) / normal[2]

# # Shift the plane to be centered around the midpoint
# xx += midpoint[0]
# yy += midpoint[1]
# zz += midpoint[2]

# Define the plane size and resolution
plane_size = 10  # Half the size of the square plane (adjust for visualization)
plane_resolution = 10  # Number of points along each axis of the plane

# Normal vector is the AHA direction
normal = average_hax / np.linalg.norm(average_hax)  # Ensure the normal vector is normalized

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


#plot
fig = plt.figure()
axis_scale = 20  
ax = fig.add_subplot(111, projection='3d') 

for i in range(len(hax_small)):

    start = p_small[i]
    end = p_small[i] + transformed_hax_small[i]*axis_scale
    
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-')
    ax.scatter(p_small[i][0], p_small[i][1], p_small[i][2], color='b', s=5)
    
    ax.scatter(translation_1_list_small[i][0], translation_1_list_small[i][1], translation_1_list_small[i][2], color='k')
    ax.scatter(translation_2_list_small[i][0], translation_2_list_small[i][1], translation_2_list_small[i][2], color='g')
    
ax.scatter(translation_2_list_small[0][0], translation_2_list_small[0][1], translation_2_list_small[0][2], color='k', s=50) 

#this is to align the plotting reference frame with the Polhemus transmitter reference frame (needed if data were acquired using the default reference system)
ax.view_init(elev=180) 

# Add the transformed average helical axis to the plot
ax.plot([transformed_average_start[0], transformed_average_end[0]],
        [transformed_average_start[1], transformed_average_end[1]],
        [transformed_average_start[2], transformed_average_end[2]], 'b-', linewidth=2, label='Average Helical Axis')

ax.scatter(transformed_average_start[0], transformed_average_start[1], transformed_average_start[2], color='b', s=100, label='AHA Position')

ax.scatter(midpoint[0], midpoint[1], midpoint[2], color='r', s=100, label='Perpendicular Plane Midpoint')

ax.plot_surface(xx, yy, zz, alpha=0.5, color='cyan', edgecolor='none', label='Perpendicular Plane')


# Update the legend to include the plane
ax.legend(['FHA', 'FHA Position', 'Sensor 1', 'Sensor 2', 'AHA', 'Perpendicular Plane'])


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
ax.legend(['FHA', 'FHA Position', 'Sensor 1', 'Sensor 2'])

plt.show()