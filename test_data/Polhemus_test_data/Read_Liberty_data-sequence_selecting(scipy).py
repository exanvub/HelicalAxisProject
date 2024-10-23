import math
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

# path = '/Users/nicolas/Desktop/Polhemus DATA/Data Bregje/'
# filename = 'Bregje-Lsnapshot.csv'
# data = path + filename

data = 'test_data/Polhemus_test_data/1_90deg_ydata.csv'

##### conversion to euler angles #####

quat_data = []
Plot_euler_sensor_1 = []
Plot_euler_sensor_2 = []
Diff_plot = []
Diff_plot_inv = []
timing = []
euler_sequence = 'xyz'

def quaternion_to_euler1(q1):
    r = R.from_quat(q1)
    return r.as_euler(euler_sequence, degrees=True)

def quaternion_to_euler2(q2):
    r = R.from_quat(q2)
    return r.as_euler(euler_sequence, degrees=True)

def calculate_angular_difference(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r = r1 * r2.inv()
    return r.as_euler(euler_sequence, degrees=True)

def calculate_angular_difference_inv(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r = r2 * r1.inv()
    return r.as_euler(euler_sequence, degrees=True)

def set_euler_sequence(sequence):
    global euler_sequence
    euler_sequence = sequence
    update_plot()


df = pd.read_csv(data)
# df = df.fillna(method='ffill') # replace all NA/NaN values in your DataFrame with the value from the previous row
# df = df.dropna()  # remove rows with missing data

def set_euler_sequence(sequence):
    global euler_sequence
    euler_sequence = sequence

    # Clear existing data
    Plot_euler_sensor_1.clear()
    Plot_euler_sensor_2.clear()
    Diff_plot.clear()
    Diff_plot_inv.clear()
    timing.clear()

    timestamp = 0
    dataout = []

    for _, row in df.iterrows():
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
        # q1 = [x1, y1, z1, w1] # correct order for scipy
        # q2 = [x2, y2, z2, w2] # correct order for scipy

        Sensor1 = quaternion_to_euler1(scipy_q1)
        Sensor2 = quaternion_to_euler2(scipy_q2)
        Plot_euler_sensor_1.append(Sensor1)
        Plot_euler_sensor_2.append(Sensor2)

        diff = calculate_angular_difference(scipy_q1, scipy_q2)
        diff_inv = calculate_angular_difference_inv(scipy_q1, scipy_q2)

        Diff_plot.append(diff)
        Diff_plot_inv.append(diff_inv)

        timing.append(timestamp)
        timestamp += 1

        dataout_row = timestamp, Sensor1[0], Sensor1[1], Sensor1[2], Sensor2[0], Sensor2[1], Sensor2[2], diff[0], diff[1], diff[2], diff_inv[0], diff_inv[1], diff_inv[2]
        dataout.append(dataout_row)

    # Update the plot after recalculating with the new sequence
    update_plot()


#### write to csv ####

# def write_data_to_csv(data, path, original_filename):
#     adjusted_filename = original_filename.replace('.csv', '_xyz-sequence_without-interpolation.csv')
#     adjusted_filepath = os.path.join(path, adjusted_filename)
#     header_row = ['time', 'X_1', 'Y_1', 'Z_1', 'X_2', 'Y_2', 'Z_2', 'diff_x', 'diff_y', 'diff_z', 'diff_x_inv', 'diff_y_inv', 'diff_z_inv']
#     with open(adjusted_filepath, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(header_row)
#         writer.writerows(data)

# write_data_to_csv(dataout, path, filename)

def update_plot():
    ax1.clear()
    ax1.plot(Plot_euler_sensor_1)
    ax1.legend([f'{axis}' for i, axis in enumerate(euler_sequence)], bbox_to_anchor=(1, 1), loc='upper left')
    ax1.set_ylabel('Sensor 1 Euler')

    ax2.clear()
    ax2.plot(Plot_euler_sensor_2)
    ax2.legend([f'{axis}' for i, axis in enumerate(euler_sequence)], bbox_to_anchor=(1, 1), loc='upper left')
    ax2.set_ylabel('Sensor 2 Euler')

    ax3.clear()
    ax3.plot(Diff_plot)
    ax3.legend([f'{axis}' for axis in euler_sequence], bbox_to_anchor=(1, 1), loc='upper left')
    ax3.set_ylabel('tibia rel. to femur')

    ax4.clear()
    ax4.plot(Diff_plot_inv)
    ax4.legend([f'{axis}_i' for axis in euler_sequence], bbox_to_anchor=(1, 1), loc='upper left')
    ax4.set_ylabel('femur rel. to tibia')

    ax5.clear()
    ax5.plot(timing)
    ax5.set_ylabel('time (s)')

    canvas.draw()



# Create GUI
root = tk.Tk()
root.title("Polhemus Data Recorder and Plotter")

# Create a frame for the plot
plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.TOP)

fig = Figure(figsize=(10, 4), dpi=100)

ax1 = fig.add_subplot(511)
ax1.plot(Plot_euler_sensor_1)
ax1.legend(['x', 'y', 'z'], bbox_to_anchor=(1, 1), loc='upper left')
ax1.set_ylabel('Sensor 1')

ax2 = fig.add_subplot(512)
ax2.plot(Plot_euler_sensor_2)
ax2.legend(['x', 'y', 'z'], bbox_to_anchor=(1, 1), loc='upper left')
ax2.set_ylabel('Sensor 2')

ax3 = fig.add_subplot(513)
ax3.plot(Diff_plot)
ax3.legend(['x', 'y', 'z'], bbox_to_anchor=(1, 1), loc='upper left')
ax3.set_ylabel('2 rel. to 1')

ax4 = fig.add_subplot(514)
ax4.plot(Diff_plot_inv)
ax4.legend(['xi', 'yi', 'zi'], bbox_to_anchor=(1, 1), loc='upper left')
ax4.set_ylabel('1 rel. to 2')

# plot timing
ax5 = fig.add_subplot(515)
ax5.plot(timing)
ax5.set_ylabel('time (s)')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
canvas.draw()


# Create a frame for the buttons
# Create radio buttons for Euler sequence selection
euler_label = tk.Label(root, text="Euler Sequence:")
euler_label.pack(side=tk.LEFT)

euler_frame = tk.Frame(root)
euler_frame.pack(side=tk.LEFT)

sequence_var = tk.StringVar()

# Create a grid to organize the radio buttons
row_num = 0
col_num = 0

xyz_radio = tk.Radiobutton(euler_frame, text="xyz", variable=sequence_var, value="xyz", command=lambda: set_euler_sequence("xyz"))
xyz_radio.grid(row=row_num, column=col_num, sticky="w")

yxz_radio = tk.Radiobutton(euler_frame, text="yxz", variable=sequence_var, value="yxz", command=lambda: set_euler_sequence("yxz"))
yxz_radio.grid(row=row_num, column=col_num+1, sticky="w")

zxy_radio = tk.Radiobutton(euler_frame, text="zyx", variable=sequence_var, value="zyx", command=lambda: set_euler_sequence("zyx"))
zxy_radio.grid(row=row_num, column=col_num+2, sticky="w")

row_num = 1

zyx_radio = tk.Radiobutton(euler_frame, text="xzy", variable=sequence_var, value="xzy", command=lambda: set_euler_sequence("xzy"))
zyx_radio.grid(row=row_num, column=col_num, sticky="w")

xzy_radio = tk.Radiobutton(euler_frame, text="yzx", variable=sequence_var, value="yzx", command=lambda: set_euler_sequence("yzx"))
xzy_radio.grid(row=row_num, column=col_num+1, sticky="w")

yzx_radio = tk.Radiobutton(euler_frame, text="zxy", variable=sequence_var, value="zxy", command=lambda: set_euler_sequence("zxy"))
yzx_radio.grid(row=row_num, column=col_num+2, sticky="w")

# Set default Euler sequence
sequence_var.set(euler_sequence)
set_euler_sequence(sequence_var.get())

update_plot()

root.mainloop()