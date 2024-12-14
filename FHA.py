
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R
import math

plt.close('all')

#### Load data ####
data = pd.read_csv('test_data/Polhemus_test_data/1_90deg_ydata_oneway.csv')
# data = pd.read_csv('test_data/Polhemus_test_data/3_90deg_y_with_xxdeg_x_rotationdata.csv')
# data= pd.read_csv('test_data/Polhemus_test_data/11_Head_rotation_2data.csv')


#### Choose method type ####
# method_type = 'all_FHA'
# method_type = 'incremental_time' 
# method_type = 'step_angle'
method_type = 'incremental_angle'


##### Define value of steps and the number of samples to skip for plotting #####
if method_type == 'all_FHA':
    step = None # Not used
    nn = 20
elif method_type == 'incremental_time':
    step = 100 # amount of samples to skip
    nn = 10
elif method_type == 'step_angle':
    step = 2 # amount of degrees to skip
    nn = 1
elif method_type == 'incremental_angle':
    step = 6 # amount of degrees to increment
    nn = 20

# Extract relevant data
time = data['Time'].values
q1 = data[['w1', 'x1', 'y1', 'z1']].values
q2 = data[['w2', 'x2', 'y2', 'z2']].values
loc1 = data[['loc1_x', 'loc1_y', 'loc1_z']].values
loc2 = data[['loc2_x', 'loc2_y', 'loc2_z']].values

###### Choose the range of data ######
cut1=200
cut2=1200
q1=q1[cut1:cut2]
q2=q2[cut1:cut2]
loc1=loc1[cut1:cut2]
loc2=loc2[cut1:cut2]
time=time[cut1:cut2]


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

all_hax = []
all_angles = []
all_svec = []
all_d = []

all_translation_1_list = []
all_translation_2_list = []


hax = []
ang = []
svec = []
d = []

translation_1_list = []
translation_2_list = []

# Calculate all helical axis:
for i in range(len(Trel)-1):
    phi, n, t, s = calculate_FHA(Trel[i], Trel[i+1])
    all_hax.append(n)
    # ang.append(phi)
    # all_angles.append(phi)
    all_angles.append(math.degrees(phi))


    all_svec.append(s)
    all_d.append(t)
    
    all_translation_1_list.append(loc1[i])
    all_translation_2_list.append(loc2[i])

def activate_method(method_type, time, Trel, loc1, loc2, all_angles):
    hax, ang, svec, d, translation_1_list, translation_2_list = [], [], [], [], [], []
    ind_incr, ind_step = [], []

    if method_type == 'all_FHA':
        hax = all_hax
        ang = all_angles
        svec = all_svec
        d = all_d

        translation_1_list = all_translation_1_list
        translation_2_list = all_translation_2_list

    elif method_type == 'incremental_time':
        # Incremental method (time-based)
        for i in range(len(time) - step):
            phi, n, t, s = calculate_FHA(Trel[i], Trel[i + step])
            hax.append(n)
            ang.append(phi)
            svec.append(s)
            d.append(t)
            
            translation_1_list.append(loc1[i])
            translation_2_list.append(loc2[i])
    
    elif method_type == 'step_angle':
        # Incremental method (step-based)
        angSum = 0
        ind_step = [0]
        for i in range(len(all_angles)):
            angSum += all_angles[i]
            if angSum > step:
                ind_step.append(i)
                angSum = 0  # Reset only after adding an index

        for i in range(1, len(ind_step)):
            phi, n, t, s = calculate_FHA(Trel[ind_step[i-1]], Trel[ind_step[i]])
            hax.append(n)
            ang.append(phi)
            svec.append(s)
            d.append(t)

            translation_1_list.append(loc1[ind_step[i-1]])
            translation_2_list.append(loc2[ind_step[i-1]])

    elif method_type == 'incremental_angle':
        # Incremental method (angle-based)
        angSum = 0
        ind_incr = []
        for i in range(len(all_angles) - 1):
            angSum = 0
            for j in range(i, len(all_angles)):
                angSum += all_angles[j]
                if angSum > step:
                    ind_incr.append([i, j])
                    break

        for i in range(1, len(ind_incr)):
            phi, n, t, s = calculate_FHA(Trel[ind_incr[i][0]], Trel[ind_incr[i][1]])
            hax.append(n)
            ang.append(phi)
            svec.append(s)
            d.append(t)

            translation_1_list.append(loc1[ind_incr[i][0]])
            translation_2_list.append(loc2[ind_incr[i][0]])

    else:
        raise ValueError("Invalid method type. Choose from 'all_FHA', 'incremental_time', 'step_angle', or 'incremental_angle'.")
    
    return hax, ang, svec, d, translation_1_list, translation_2_list, ind_incr, ind_step


hax, ang, svec, d, translation_1_list, translation_2_list, ind_incr, ind_step = activate_method(method_type, time, Trel, loc1, loc2, all_angles)


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
    
ax.scatter(translation_2_list_small[0][0], translation_2_list_small[0][1], translation_2_list_small[0][2], color='k', s=10) 

#this is to align the plotting reference frame with the Polhemus transmitter reference frame (needed if data were acquired using the default reference system)
ax.view_init(elev=180) 

# Add the transformed average helical axis to the plot
ax.plot([transformed_average_start[0], transformed_average_end[0]],
        [transformed_average_start[1], transformed_average_end[1]],
        [transformed_average_start[2], transformed_average_end[2]], 'b-', linewidth=2, label='Average Helical Axis')

ax.scatter(transformed_average_start[0], transformed_average_start[1], transformed_average_start[2], color='b', s=10, label='AHA Position')

ax.scatter(midpoint[0], midpoint[1], midpoint[2], color='r', s=10, label='Perpendicular Plane Midpoint')

ax.plot_surface(xx, yy, zz, alpha=0.5, color='cyan', edgecolor='none', label='Perpendicular Plane')



# Plot intersection points
ax.scatter(intersections_points_small[:, 0], intersections_points_small[:, 1], intersections_points_small[:, 2], color='magenta', label='Intersections')

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

plt.title(method_type)

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
plt.figure()
plt.scatter(plane_points_2d[:, 0], plane_points_2d[:, 1], c='b', label='FHA Intersection Points')
plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
plt.axvline(0, color='k', linestyle='--', linewidth=0.8)
plt.grid(True, linestyle=':')
plt.xlabel('U (Plane Axis 1)')
plt.ylabel('V (Plane Axis 2)')
plt.title('FHA Intersection Points in Plane Coordinate System')
plt.legend()
plt.show()