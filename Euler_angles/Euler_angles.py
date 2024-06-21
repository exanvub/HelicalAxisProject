##############################################################################################################
#### enter file path and name:

#path = './test_data/Polhemus_test_data/'
#filename = 'Polhemus_90degZ_1data.csv'

#path = './test_data/IMU_test_data/'
#filename = '90deg_x.csv'

path = '/Users/nicolas/Desktop/'
filename = 'right_knee_endo-exo_in_ext-flex_example.csv'
##############################################################################################################

import math
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


###### conversion to euler angles ######

quat_data = []
Plot_euler_sensor_1 = []
Plot_euler_sensor_2 = []
Diff_yzx_plot = []
Diff_yxz_plot = []
Diff_xyz_plot = []
Diff_xzy_plot = []
Diff_xyz_plot_inv = []

def quaternion_to_euler1(w1, x1, y1, z1):
    # Normalize the quaternion
    magnitude = math.sqrt(w1 * w1 + x1 * x1 + y1 * y1 + z1 * z1)
    w1 /= magnitude
    x1 /= magnitude
    y1 /= magnitude
    z1 /= magnitude

    # # XYZ sequence
    # X_1 = math.atan2(2 * (w1 * x1 + y1 * z1), 1 - 2 * (x1 * x1 + y1 * y1))
    # Y_1 = math.asin(2 * (w1 * y1 - z1 * x1))
    # Z_1 = math.atan2(2 * (w1 * z1 + x1 * y1), 1 - 2 * (y1 * y1 + z1 * z1))

    # # XZY sequence
    # X_1 = math.atan2(2 * (w1 * x1 - y1 * z1), 1 - 2 * (x1 * x1 + z1 * z1))
    # Z_1 = math.asin(2 * (w1 * y1 + x1 * z1))
    # Y_1 = math.atan2(2 * (w1 * z1 + y1 * x1), 1 - 2 * (y1 * y1 + z1 * z1))

    # ZYX sequence
    Z_1 = math.atan2(2 * (w1 * x1 - z1 * y1), 1 - 2 * (x1 * x1 + y1 * y1))
    Y_1 = math.asin(2 * (w1 * y1 + z1 * x1))
    X_1 = math.atan2(2 * (w1 * z1 + x1 * y1), 1 - 2 * (y1 * y1 + z1 * z1))

    X_1 = math.degrees(X_1)
    Y_1 = math.degrees(Y_1)
    Z_1 = math.degrees(Z_1)

    return X_1, Y_1, Z_1

def quaternion_to_euler2(w2, x2, y2, z2):
#### for sensor 2

    magnitude = math.sqrt(w2 * w2 + x2 * x2 + y2 * y2 + z2 * z2)
    w2 /= magnitude
    x2 /= magnitude
    y2 /= magnitude
    z2 /= magnitude

    # # XZY sequence
    # X_2 = math.atan2(2 * (w2 * x2 - y2 * z2), 1 - 2 * (x2 * x2 + z2 * z2))
    # Z_2 = math.asin(2 * (w2 * y2 + x2 * z2))
    # Y_2 = math.atan2(2 * (w2 * z2 + y2 * x2), 1 - 2 * (y2 * y2 + z2 * z2))

    # # ZXY sequence
    # Z_2 = math.atan2(2 * (w2 * x2 + y2 * z2), 1 - 2 * (x2 * x2 + y2 * y2))
    # X_2 = math.asin(2 * (w2 * y2 - x2 * z2))
    # Y_2 = math.atan2(2 * (w2 * z2 - x2 * y2), 1 - 2 * (y2 * y2 + z2 * z2))

    # # XYZ sequence
    # X_2 = math.atan2(2 * (w2 * x2 + y2 * z2), 1 - 2 * (x2 * x2 + y2 * y2))
    # Y_2 = math.asin(2 * (w2 * y2 - z2 * x2))
    # Z_2 = math.atan2(2 * (w2 * z2 + x2 * y2), 1 - 2 * (y2 * y2 + z2 * z2))

    # # YXZ sequence
    # Y_2 = math.atan2(2 * (w2 * x2 + z2 * y2), 1 - 2 * (x2 * x2 + z2 * z2))
    # X_2 = math.asin(2 * (w2 * y2 - x2 * z2))
    # Z_2 = math.atan2(2 * (w2 * z2 + y2 * x2), 1 - 2 * (x2 * x2 + y2 * y2))

    # # YZX sequence
    # Y_2 = math.atan2(2 * (w2 * x2 - z2 * y2), 1 - 2 * (x2 * x2 + y2 * y2))
    # Z_2 = math.asin(2 * (w2 * y2 + z2 * x2))
    # X_2 = math.atan2(2 * (w2 * z2 + x2 * y2), 1 - 2 * (x2 * x2 + z2 * z2))

    # ZYX sequence
    Z_2 = math.atan2(2 * (w2 * x2 - z2 * y2), 1 - 2 * (x2 * x2 + y2 * y2))
    Y_2 = 2 * (w2 * y2 + z2 * x2)
    Y_2 = +1.0 if Y_2 > +1.0 else Y_2
    Y_2 = -1.0 if Y_2 < -1.0 else Y_2
    Y_2 = math.asin(Y_2)
    X_2 = math.atan2(2 * (w2 * z2 + x2 * y2), 1 - 2 * (y2 * y2 + z2 * z2))

    X_2 = math.degrees(X_2)
    Y_2 = math.degrees(Y_2)
    Z_2 = math.degrees(Z_2)

    return X_2, Y_2, Z_2

def calculate_angular_difference_xzy(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    # r = r2 * r1.inv()
    r = r1 * r2.inv() #q2 relative to q1
    return r.as_euler('xzy', degrees=True)

def calculate_angular_difference_xyz_inv(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r = r2 * r1.inv() #q1 relative to q2
    # r = r1 * r2.inv()
    return r.as_euler('xyz', degrees=True)

def calculate_angular_difference_yxz(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    # r = r2 * r1.inv()
    r = r1 * r2.inv()
    return r.as_euler('yxz', degrees=True)

def calculate_angular_difference_xyz(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    # r = r2 * r1.inv()
    r = r1 * r2.inv()
    return r.as_euler('xyz', degrees=True)


##### create plots #####

t=0
dataout = []


with open(path+filename, 'r') as csvfile:
    data = csv.DictReader(csvfile)
    #fieldnames = ['Time','w1','x1','y1','z1','w2','x2','y2','z2','x_diff','y_diff','z_diff']

    for row in data:
        w1 = float(row['w1'])
        x1 = float(row['x1'])
        y1 = float(row['y1'])
        z1 = float(row['z1'])
        w2 = float(row['w2'])
        x2 = float(row['x2'])
        y2 = float(row['y2'])
        z2 = float(row['z2'])
        q1 = w1, x1, y1, z1
        q2 = w2, x2, y2, z2
        scipy_q1 = x1, y1, z1, w1
        scipy_q2 = x2, y2, z2, w2

        
        X_1, Y_1, Z_1 = quaternion_to_euler1(w1, x1, y1, z1)
        X_2, Y_2, Z_2 = quaternion_to_euler2(w2, x2, y2, z2)
        Plot_euler_sensor_1.append([X_1, Y_1, Z_1])
        Plot_euler_sensor_2.append([X_2, Y_2, Z_2])

        diff_xzy = calculate_angular_difference_xzy(scipy_q1, scipy_q2)
        diff_xyz_inv = calculate_angular_difference_xyz_inv(scipy_q1, scipy_q2)
        diff_yxz = calculate_angular_difference_yxz(scipy_q1, scipy_q2)
        diff_xyz = calculate_angular_difference_xyz(scipy_q1, scipy_q2)

        Diff_xzy_plot.append(diff_xzy)
        Diff_yxz_plot.append(diff_yxz)
        Diff_xyz_plot.append(diff_xyz)
        Diff_xyz_plot_inv.append(diff_xyz_inv)

        #output for the csv file (time, euler angles, angular difference) adjust changes in the header row accordingly (see below)
        t = t + 1
        dataout_row = t, X_1, Y_1, Z_1, X_2, Y_2, Z_2,diff_xyz[0], diff_xyz[1], diff_xyz[2] ,diff_xyz_inv[0], diff_xyz_inv[1], diff_xyz_inv[2]
        dataout.append(dataout_row)


fig = plt.figure(figsize=(8, 6), dpi=100)
plt.title('x=flex/ext, y=valgus/varus, z=int/ext rot')

ax1 = fig.add_subplot(411)
ax1.plot(Plot_euler_sensor_1)
ax1.legend(['x', 'y', 'z'], bbox_to_anchor=(1, 1), loc='upper left')
ax1.set_ylabel('Sensor 1 Euler (zyx)')

ax2 = fig.add_subplot(412)
ax2.plot(Plot_euler_sensor_2)
ax2.legend(['x', 'y', 'z'], bbox_to_anchor=(1, 1), loc='upper left')
ax2.set_ylabel('Sensor 2 Euler (zyx)')

ax3 = fig.add_subplot(413)
ax3.plot(Diff_xyz_plot)
ax3.legend(['x', 'y', 'z'], bbox_to_anchor=(1, 1), loc='upper left')
ax3.set_ylabel('tibia rel. to femur')

ax4 = fig.add_subplot(414)
ax4.plot(Diff_xyz_plot_inv)
ax4.legend(['xi', 'yi', 'zi'], bbox_to_anchor=(1, 1), loc='upper left')
ax4.set_ylabel('femur rel. to tibia')

plt.show()

# #### write to csv ####
# import os
# def write_data_to_csv(data,path, original_filename):
#     adjusted_filename = original_filename.replace('.csv', '_euler-xyz-sequence.csv')
#     adjusted_filepath = os.path.join(path, adjusted_filename)
#     #header_row = ['time', 'q1_w', 'q1_x', 'q1_y', 'q1_z', 'q2_w', 'q2_x', 'q2_y', 'q2_z'] #Quat only
#     header_row = ['time', 'X_1', 'Y_1', 'Z_1', 'X_2', 'Y_2', 'Z_2', 'diff_x', 'diff_y', 'diff_z', 'diff_x_inv', 'diff_y_inv', 'diff_z_inv']
#     with open(adjusted_filepath, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(header_row)
#         writer.writerows(data)

# write_data_to_csv(dataout,path, filename)

