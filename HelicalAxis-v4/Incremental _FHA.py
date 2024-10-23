import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.transform import Rotation as R

data = pd.read_csv('test_data/Polhemus_test_data/1_90deg_ydata_oneway.csv')

# # drop the first n rows
# data = data.drop(data.index[0:400])

# Extract relevant data
time = data['Time'].values
q1 = data[['x1', 'y1', 'z1', 'w1']].values
q2 = data[['x2', 'y2', 'z2', 'w2']].values
loc1 = data[['loc1_x', 'loc1_y', 'loc1_z']].values
loc2 = data[['loc2_x', 'loc2_y', 'loc2_z']].values

def calculate_quat_diff(quat1, quat2):
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    r = r1 * r2.inv()
    return r.as_quat()

# convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(quaternion):
    r = R.from_quat(quaternion)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

# calculate rotation angle from the rotation matrix
def calculate_angle(rotation_matrix):
    # Trace of the rotation matrix
    trace = np.trace(rotation_matrix)
    # Angle of rotation
    angle = np.arccos((trace - 1) / 2)
    return angle

# calculate the finite helical axis
def finite_helical_axis(quat_pose_1, quat_pose_2, translation_pose_1, translation_pose_2):
    # Convert the quaternions to rotation matrices
    R1 = quaternion_to_rotation_matrix(quat_pose_1)
    R2 = quaternion_to_rotation_matrix(quat_pose_2)
    
    # Calculate relative rotation matrix between the two poses
    R_rel = np.dot(np.linalg.inv(R1), R2)
    
    # Calculate the angle of rotation
    angle = calculate_angle(R_rel)
    
    # Find the rotation axis (finite helical axis)
    axis = np.array([R_rel[2, 1] - R_rel[1, 2], 
                     R_rel[0, 2] - R_rel[2, 0], 
                     R_rel[1, 0] - R_rel[0, 1]]) / (2 * np.sin(angle))
    
    # Relative translation between the two poses
    translation_rel = translation_pose_2 - translation_pose_1
    
    # Calculate the position of the finite helical axis
    position = np.cross(axis, translation_rel) / (2 * np.sin(angle))
    
    return axis, angle, position


quat_diff_list = []
translation_diff_list = []
axis_list = []
angle_list = []
position_list = []
translation_1_list = []
translation_2_list = []


# Loop with step size of n to skip rows between poses
n = 2
for i in range(len(time) - n):  # Ensure the loop doesn't go out of bounds
    quat1 = q1[i]
    quat2 = q2[i]
    
    # Get the pose n steps ahead
    quat1_next = q1[i + n]
    quat2_next = q2[i + n]
    
    # quaternion difference between current pose and the one n steps ahead
    quat_diff = calculate_quat_diff(quat1, quat2)
    quat_diff_next = calculate_quat_diff(quat1_next, quat2_next)

    
    # translations for the current pose and the one n steps ahead
    translation1 = loc1[i]
    translation2 = loc2[i]
    translation1_next = loc1[i + n]
    translation2_next = loc2[i + n]
    
    # translation difference between the current pose and the one n steps ahead
    translation_diff = translation2 - translation1
    translation_diff_next = translation2_next - translation1_next
    

    axis, angle, position = finite_helical_axis(quat_diff, quat_diff_next, translation_diff, translation_diff_next)
    
    axis_list.append(axis)
    angle_list.append(angle)
    position_list.append(position)
    translation_1_list.append(translation1)
    translation_2_list.append(translation2)
    

# create a list with only every nn-th element, when plotting computer says no :D 
nn = 50

axis_list_small = axis_list[::nn]
position_list_small = position_list[::nn]
translation_1_list_small = translation_1_list[::nn]
translation_2_list_small = translation_2_list[::nn]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

axis_scale = 20  # adjustable for visualization

for i in range(len(axis_list_small)):
    axis = axis_list_small[i]
    position = position_list_small[i]
    
    # Start and end points of the finite helical axis
    start = position
    end = position + axis * axis_scale  # Extend the axis for visualization

    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-')
    
    ax.scatter(position[0], position[1], position[2], color='b', s=5)

    ax.scatter(translation_1_list_small[i][0], translation_1_list_small[i][1], translation_1_list_small[i][2], color='k')
    ax.scatter(translation_2_list_small[i][0], translation_2_list_small[i][1], translation_2_list_small[i][2], color='g')

# Labeling axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

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

# add a legend
ax.legend(['FHA', 'FHA Position', 'Sensor 1', 'Sensor 2'])


# Show the plot
plt.show()

# convert angle_list to degrees
angle_list = np.degrees(angle_list)
# plot angle
plt.plot(angle_list)
plt.show()