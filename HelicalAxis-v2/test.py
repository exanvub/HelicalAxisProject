import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def quaternionToMatrix(q):
    """
    Convert a quaternion to a rotation matrix.
    """
    q0, q1, q2, q3 = q
    R = np.array([[1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                  [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
                  [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]])
    return R

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

# Define the function computeFHAworld (assuming you have defined matrixVectorToHA)
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
        R_ = R_post[i] @ R_pre[i].T
        v_ = v_post[i] - (R_ @ v_pre[i])

        # calculate this timestep
        n[i], r0[i], phi[i], l[i] = matrixVectorToHA(R_, v_)  # matrixVectorToHA is not defined here

        # compute alternative locations for r0
        r0_displ_tar[i]  = np.dot(n[i], v_pre[i])

    return n, r0, r0_displ_base, r0_displ_tar, phi, l

# Function to read the CSV file and process the data
def process_csv_file(filename):
    # Read the CSV file
    data = pd.read_csv(filename)

    # only use every 5th row
    data = data.iloc[::2, :]

    # Extract relevant columns
    time = data[['Time']]
    quat_wxyz = data[['w1', 'x1', 'y1', 'z1']]
    translation = data[['loc1_x', 'loc1_y', 'loc1_z']] # Assuming the translation columns are named loc1_x, loc1_y, loc1_z
    # translation = np.zeros((len(data), 3)) # Replace with the actual translation data

    # Convert data to numpy arrays
    time_np = time.values
    quat_wxyz_np = quat_wxyz.values
    translation_np = translation.values

    # Reshape the quaternion and translation arrays
    quat_wxyz_np = quat_wxyz_np.reshape((-1, 4))
    translation_np = translation_np.reshape((-1, 3))

    return time_np, quat_wxyz_np, translation_np

# Load data from CSV file
filename = 'Polhemus_90degYdata.csv'  # Replace with the actual filename
time, quat_wxyz, translation = process_csv_file(filename)

# Convert quaternion to rotation matrices
rotation_matrices = [R.from_quat(q).as_matrix() for q in quat_wxyz]

# Apply computeFHAworld function
n, r0, r0_displ_base, r0_displ_tar, phi, l = computeFHAworld(np.array(rotation_matrices), translation)

# Plot the normal vectors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Plot normal vectors
# for i in r0:
#     ax.quiver(0, 0, 0, i[0], i[1], i[2])

for normal in n:
    ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], color='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Normal Vectors')
plt.show()

# Print or use the results as needed
print("Normal Vectors:", n)
print("Support Vectors:", r0)
print("Base Displacement:", r0_displ_base)
print("Target Displacement:", r0_displ_tar)
print("Rotation Angles:", phi)
print("Absolute Translations:", l)
