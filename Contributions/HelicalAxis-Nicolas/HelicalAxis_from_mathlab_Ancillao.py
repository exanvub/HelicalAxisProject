import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the .mat file
data = scipy.io.loadmat('./HelicalAxis-Literature/Andrea Ancillao/IHA_paper-main/DATA/HingeSampleData.mat')
hinge = data['hinge']


# Extract relevant data
T1 = hinge['T1_0'][0][0]
T2 = hinge['T2_0'][0][0]
GA_0 = hinge['GA_0'][0][0]
markers = hinge['markers'][0][0]
tw21_1 = hinge['tw21_1'][0][0]

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
def calcIHA(tw):
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
nF = T1.shape[2]
for k in range(0, 200, 5):
    plot_frame(ax, T1[:, :, k], 'b')
    plot_frame(ax, T2[:, :, k], 'r')

# Adjusting axes limits
all_points = np.concatenate([T1[:, :, :], T2[:, :, :]], axis=-1)
mins = np.min(all_points[:3, :, :], axis=(1, 2))
maxs = np.max(all_points[:3, :, :], axis=(1, 2))
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
IHA, AHA = calcIHA(tw21_1)

# Plot the motion in the coordinate system {0}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.grid(True)
ax.set_box_aspect([1, 1, 1])

# Plot the IHA
# for k in range(IHA['n'].shape[0]):
#     origin = IHA['s'][k]
#     direction = IHA['n'][k]
#     ax.quiver(origin[0], origin[1], origin[2], direction[0], direction[1], direction[2], color='g')

# Plot the IHA as lines
# Calculate maximum range of quiver vectors
max_range = np.nanmax(np.linalg.norm(IHA['n'], axis=1))

for k in range(IHA['n'].shape[0]):
    origin = IHA['s'][k]
    direction = IHA['n'][k]
    end_point = origin + direction * max_range  # Extend the line for better visibility
    ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2]], color='g')

# Plot the AHA as lines
origin_AHA = AHA['s']
direction_AHA = AHA['n']
end_point_AHA = origin_AHA + direction_AHA * max_range  # Extend the line for better visibility
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