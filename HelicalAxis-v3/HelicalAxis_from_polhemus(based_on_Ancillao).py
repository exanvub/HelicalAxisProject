import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Read the CSV file containing sensor data
# data = pd.read_csv('./test_data/Polhemus_test_data/Polhemus_90degX(clean)_2data.csv')
# data = pd.read_csv('./test_data/Polhemus_test_data/Polhemus_90degY_2data.csv')
data = pd.read_csv('/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/1_90deg_ydata.csv')


# Extract relevant data
time = data['Time'].values
quat1 = data[['x1', 'y1', 'z1', 'w1']].values
quat2 = data[['x2', 'y2', 'z2', 'w2']].values
loc1 = data[['loc1_x', 'loc1_y', 'loc1_z']].values
loc2 = data[['loc2_x', 'loc2_y', 'loc2_z']].values

# convert quaternions to euler angles
euler1 = R.from_quat(quat1).as_euler('xyz', degrees=True)
euler2 = R.from_quat(quat2).as_euler('xyz', degrees=True)

# calculate euler differences between the two sensors from the quaternions
r1 = R.from_quat(quat1)
r2 = R.from_quat(quat2)
r = r2 * r1.inv()
euler_diff = r.as_euler('xyz', degrees=True)

# # plot the euler angles
# plt.figure()
# plt.plot(time, euler1[:, 0], label='x 1')
# plt.plot(time, euler1[:, 1], label='y 1')
# plt.plot(time, euler1[:, 2], label='z 1')
# plt.plot(time, euler2[:, 0], label='x 2')
# plt.plot(time, euler2[:, 1], label='y 2')
# plt.plot(time, euler2[:, 2], label='z 2')
# plt.xlabel('Time [s]')
# plt.ylabel('Angle [deg]')
# plt.title('Euler angles')
# plt.legend()
# plt.show()

# plot the euler differences
plt.figure()
plt.plot(time, euler_diff[:, 0], label='x')
plt.plot(time, euler_diff[:, 1], label='y')
plt.plot(time, euler_diff[:, 2], label='z')
plt.xlabel('Time [s]')
plt.ylabel('Angle [deg]')
plt.title('Euler angle differences')
plt.legend()
plt.show()


# Convert quaternions to rotation matrices
rot1 = R.from_quat(quat1).as_matrix()
rot2 = R.from_quat(quat2).as_matrix()

# Define the mvarray function
def mvarray(vec):
    magnitude = np.linalg.norm(vec, axis=1)
    unit_vec = vec / np.expand_dims(magnitude, axis=1)
    return magnitude, unit_vec

# Define the HA2CS function
def HA2CS(n, s):
    nF = n.shape[0]
    _, z = mvarray(n)  # ensure n is a unit vector
    tmp = np.cross(z, s)  # auxiliary direction

    x = np.cross(n, tmp)
    _, x = mvarray(x)
    _, y = mvarray(np.cross(z, x))
    T = np.empty((4, 4, nF))  # build matrix T
    for k in range(nF):
        O = s[k, :]
        e1 = x[k, :]
        e2 = y[k, :]
        e3 = z[k, :]
        T[:, :, k] = np.array([[e1[0], e2[0], e3[0], O[0]],
                               [e1[1], e2[1], e3[1], O[1]],
                               [e1[2], e2[2], e3[2], O[2]],
                               [0, 0, 0, 1]])
    return T

# Define the calcIHA function
def calcIHA(rot1, rot2, loc1, loc2):
    tw = np.empty((rot1.shape[0], 6))
# Calculate tw
    for i in range(rot2.shape[0]):
        tw[i, :3] = np.dot(rot2[i].T, loc1[i] - loc2[i])
        # Extract angular velocity from the rotation matrix
        angular_velocity_matrix = np.dot(rot2[i].T, rot1[i])
        angular_velocity = np.array([angular_velocity_matrix[2, 1], angular_velocity_matrix[0, 2], angular_velocity_matrix[1, 0]])
        tw[i, 3:] = angular_velocity

    nF = tw.shape[0]

    # IHA
    w = tw[:, 0:3]
    v = tw[:, 3:6]
    wmod = np.linalg.norm(w, axis=1)
    n = w / np.expand_dims(wmod, axis=1) 
    s = np.cross(w, v) / np.expand_dims(wmod ** 2, axis=1)

    # Define weights (based on w) (for optimal pivot point only)
    wg = wmod / np.nanmax(wmod)
    wg2 = LogisticFunc(wg, 1, 0.3, 30)  # logistic function

    # Apply the threshold and filter IHA
    norig = n.copy()
    sorig = s.copy()
    trsh = 0.1 * np.nanmax(wmod)  # rad/s
    for k in range(nF):
        if wmod[k] <= trsh:
            n[k, :] = np.nan
            s[k, :] = np.nan

    # LCS attached to the IHA
    T = HA2CS(n, s)

    # Store IHA data
    IHA = {'n': n, 's': s, 'T': T}

    # Optimal screw point - weighted
    n = norig.copy()
    s = sorig.copy()
    Qi = np.empty((3, 3, nF))
    si = np.empty((nF, 3))
    for i in range(nF):
        Qi[:, :, i] = wg2[i] * (np.eye(3) - np.outer(n[i, :], n[i, :]))
        si[i, :] = np.dot(Qi[:, :, i], s[i, :])
    Q = np.nanmean(Qi, axis=2)
    Q2 = np.nanmean(si, axis=0)
    Sopt = np.linalg.solve(Q, Q2)

    # Optimal direction - weighted
    ni = np.empty((3, 3, nF))
    for i in range(nF):
        ni[:, :, i] = wg2[i] * np.outer(n[i, :], n[i, :])
    ntot = np.nansum(ni, axis=2)
    U, _, _ = np.linalg.svd(ntot)
    Nopt = U[:, 0]

    # LCS attached to the AHA
    Topt = HA2CS(np.expand_dims(Nopt, axis=0), np.expand_dims(Sopt, axis=0))

    AHA = {'n': Nopt, 's': Sopt, 'T': Topt}

    return IHA, AHA

# Define the LogisticFunc function
def LogisticFunc(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))



# Plot the motion in the coordinate system {0}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.grid(True)
ax.set_box_aspect([1, 1, 1])

# Function to plot a frame
def plot_frame(ax, T, color):
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color=color)
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color=color)
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color=color)

# Plot the initial pose
nF = rot1.shape[0]

for k in range(0, nF, 50):
    T1 = np.eye(4)
    T1[:3, :3] = rot1[k]
    T1[:3, 3] = loc1[k]
    T2 = np.eye(4)
    T2[:3, :3] = rot2[k]
    T2[:3, 3] = loc2[k]
    plot_frame(ax, T1, 'b')
    plot_frame(ax, T2, 'r')

# Adjusting axes limits
all_points = np.concatenate([loc1, loc2], axis=0)
mins = np.min(all_points, axis=0)
maxs = np.max(all_points, axis=0)
ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.set_zlim(mins[2], maxs[2])

# Setting labels and title
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Motion of the hinge joint')

plt.show()

# Calculate IHA and AHA
IHA, AHA = calcIHA(rot1, rot2, loc1, loc2)

# Plot the motion in the coordinate system {0}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.grid(True)
ax.set_box_aspect([1, 1, 1])

# Plot the IHA as lines
max_range = np.nanmax(np.linalg.norm(IHA['n'], axis=1))

for k in range(IHA['n'].shape[0]):
    origin = IHA['s'][k]
    direction = IHA['n'][k]
    end_point = origin + direction * max_range  
    ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2]], color='g')


# Plot the AHA as lines
origin_AHA = AHA['s']
direction_AHA = AHA['n']
end_point_AHA = origin_AHA + direction_AHA * max_range  
ax.plot([origin_AHA[0], end_point_AHA[0]], [origin_AHA[1], end_point_AHA[1]], [origin_AHA[2], end_point_AHA[2]], color='b')


# Adjusting axes limits
ax_lim = [-max_range, max_range]
ax.set_xlim(ax_lim)
ax.set_ylim(ax_lim)
ax.set_zlim(ax_lim)

# Setting labels and title
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Instantaneous Helical Axis of the hinge motion')

plt.show()