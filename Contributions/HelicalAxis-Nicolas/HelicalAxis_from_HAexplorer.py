
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Read the CSV file containing sensor data
# data = pd.read_csv('./test_data/Polhemus_test_data/Polhemus_90degX(clean)_2data.csv')
# data = pd.read_csv('./test_data/Polhemus_test_data/Polhemus_90degY_2data.csv')
data = pd.read_csv('test_data/Polhemus_test_data/3_90deg_y_with_xxdeg_x_rotationdata.csv')

# Extract relevant data
time = data['Time'].values
quat1 = data[['x1', 'y1', 'z1', 'w1']].values
quat2 = data[['x2', 'y2', 'z2', 'w2']].values
loc1 = data[['loc1_x', 'loc1_y', 'loc1_z']].values
loc2 = data[['loc2_x', 'loc2_y', 'loc2_z']].values

# only use every 10th row
time = time[::10]
quat1 = quat1[::10]
quat2 = quat2[::10]
loc1 = loc1[::10]
loc2 = loc2[::10]

# Convert quaternions to rotation matrices
rot1 = R.from_quat(quat1).as_matrix()
rot2 = R.from_quat(quat2).as_matrix()

# # reshape the rotation matrices to 1D
# rot2_1D = rot2.reshape(-1, 9)
# #save rot2_1D to txt
# np.savetxt('rot2_1D.txt', rot2_1D)
# np.savetxt('loc2.txt', loc2)

r1 = R.from_quat(quat1)
r2 = R.from_quat(quat2)
r = r1.inv() * r2
Rdiff = r.as_matrix()


### From HAexplorer:
def computeFHAworld(R, v):
    """
    Computes the finite helical axes from a list of model transformations R, v.
    The result is relative to the world ("traditional" FHA)

    Returns four np.arrays with one element per time step:
      - n: normal vector (direction) of helical axis
      - r0: helical axis support vector closest to origin
      - phi: rotation around axis in [0,pi]
      - l: displacement length along axis (can be negative)
    """
    # shift entries
    R_pre = R[:-1,:,:]
    v_pre = v[:-1,:]
    R_post = R[1:,:,:]
    v_post = v[1:,:]

    # result arrays containing every timestep
    n =   np.zeros(v_pre.shape)
    r0 =  np.zeros(v_pre.shape)
    r0_displ_base = np.zeros(v_pre.shape[0])
    r0_displ_tar =  np.zeros(v_pre.shape[0])
    phi = np.zeros(v_pre.shape[0])
    l =   np.zeros(v_pre.shape[0])

    for i in range(v_pre.shape[0]):
        # calculate pre->post matrices
        R = R_post[i] @ R_pre[i].T
        v = v_post[i] - (R @ v_pre[i])

        # calculate this timestep
        n[i], r0[i], phi[i], l[i] = matrixVectorToHA(R, v)

        # compute alternative locations for r0
        r0_displ_tar[i]  = np.dot(n[i], v_pre[i])

    return n, r0, r0_displ_base, r0_displ_tar, phi, l

def computeFHAref(R_ref, v_ref, R, v):
    """
    Computes the finite helical axes from a list of model transformations R, v.
    The result is relative to the transformations of a reference system R_ref, v_ref.

    Returns four np.arrays with one element per time step:
      - n: normal vector (direction) of helical axis
      - r0: helical axis support vector closest to origin
      - phi: rotation around axis in [0,pi]
      - l: displacement length along axis (can be negative)
    """
    # shift entries
    R_pre = R[:-1,:,:]
    v_pre = v[:-1,:]
    R_post = R[1:,:,:]
    v_post = v[1:,:]

    R_ref_pre = R_ref[:-1,:,:]
    v_ref_pre = v_ref[:-1,:]
    R_ref_post = R_ref[1:,:,:]
    v_ref_post = v_ref[1:,:]

    # result arrays containing every timestep
    n =             np.zeros(v_pre.shape)
    r0 =            np.zeros(v_pre.shape)
    r0_displ_base = np.zeros(v_pre.shape[0])
    r0_displ_tar =  np.zeros(v_pre.shape[0])
    phi =           np.zeros(v_pre.shape[0])
    l =             np.zeros(v_pre.shape[0])

    for i in range(v_pre.shape[0]):
        # calculate pre->post matrices
        R = R_post[i] @ R_pre[i].T
        v = v_post[i] - (R @ v_pre[i])
        R_ref = R_ref_post[i] @ R_ref_pre[i].T
        v_ref = v_ref_post[i] - (R_ref @ v_ref_pre[i])

        # rotation/translation relative to the reference system
        R = R_ref.T @ R
        v = R_ref.T @ (v - v_ref)

        # calculate this timestep
        n[i], r0[i], phi[i], l[i] = matrixVectorToHA(R, v)

        # compute alternative locations for r0
        r0_displ_base[i] = np.dot(n[i], v_ref_pre[i])
        r0_displ_tar[i]  = np.dot(n[i], v_pre[i])

    return n, r0, r0_displ_base, r0_displ_tar, phi, l


def matrixVectorToHA(R, v):
    """
    Computes a single helical axis that describes the screwing
    of the rotation/translation given by R,v.
    """
    # sine, cosine of rotation angle
    sin_phi = 0.5 * np.sqrt((R[2,1]-R[1,2])**2 + (R[0,2]-R[2,0])**2 + (R[1,0]-R[0,1])**2)
    cos_phi = 0.5 * (np.trace(R) - 1)
    #sin_phi = 0.5 * np.sqrt((3-np.trace(R))*(1+np.trace(R))) # same error as cos_phi

    # use sine approximation, if sinphi <= (1/2)*sqrt(2), else use cosphi
    phi = 0.0
    if sin_phi <= 0.5*np.sqrt(2.0):
        phi = np.arcsin(sin_phi)
        if cos_phi < 0:
            phi = np.pi - phi
        cos_phi = np.cos(phi) # re-compute for numerical precision
    else:
        phi = np.arccos(cos_phi)
        sin_phi = np.sin(phi) # re-compute for numerical precision

    # helical axis
    nbar = (R - R.T) / (2*sin_phi)
    n = np.array([nbar[2,1], nbar[0,2], nbar[1,0]])

    # absolute translation along axis
    l = np.dot(n,v)

    # axis support vector
    n_cross_v = np.cross(n,v)
    r0 = -0.5*np.cross(n, n_cross_v) + sin_phi / (2.0*(1.0-cos_phi)) * n_cross_v

    return n, r0, phi, l

# Compute FHA for the extracted data
n, r0, r0_displ_base, r0_displ_tar, phi, l = computeFHAworld(rot1, loc1)
n_2, r0_2, r0_displ_base_2, r0_displ_tar_2, phi_2, l_2 = computeFHAworld(rot2, loc2)
n_ref, r0_ref, r0_displ_base_ref, r0_displ_tar_ref, phi_ref, l_ref = computeFHAref(rot1, loc1, rot2, loc2)

# Plotting results for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot helical axis
# ax.quiver(r0[:, 0], r0[:, 1], r0[:, 2], n[:, 0], n[:, 1], n[:, 2], length=0.1, normalize=True)
# ax.plot([r0[0], n[0]], [r0[1], n[1]], [r0[2], n[2]], color='g', label='Helical Axis')

#plot r0
# ax.scatter(r0[:, 0], r0[:, 1], r0[:, 2], color='g')
# ax.scatter(n[:, 0], n[:, 1], n[:, 2], color='b')
# ax.scatter(n_2[:, 0], n_2[:, 1], n_2[:, 2], color='b')
# ax.scatter(r0_2[:, 0], r0_2[:, 1], r0_2[:, 2], color='r')
#plot n
# ax.quiver(r0[:, 0], r0[:, 1], r0[:, 2], n[:, 0], n[:, 1], n[:, 2], color='g', length=1, normalize=True)

# Plot lines from points n_2 to points r0_2
for i in range(n_2.shape[0]):
    ax.plot([r0_ref[i, 0], n_ref[i, 0]], [r0_ref[i, 1], n_ref[i, 1]], [r0_ref[i, 2], n_ref[i, 2]], color='r')

# Plot original points
ax.scatter(loc1[:, 0], loc1[:, 1], loc1[:, 2], color='g', label='Sensor 1')
ax.scatter(loc2[:, 0], loc2[:, 1], loc2[:, 2], color='g', label='Sensor 2')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()