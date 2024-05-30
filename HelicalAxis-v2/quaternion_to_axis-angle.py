# #read pd from csv file
# import pandas as pd
# import numpy as np
# import math
# import csv

# #read csv file
# df = pd.read_csv('DATA/knee_kinematics/1-Ldata.csv')
# print(df)

# #convert quaternion to axis-angle
# def quaternion_to_axis_angle(q):
#     # Extract the vector part of the quaternion
#     x = q[0]
#     y = q[1]
#     z = q[2]
#     # Calculate the length of the vector part
#     magnitude = math.sqrt(x * x + y * y + z * z)
#     # If the length is close to 0, the angle is 0
#     if magnitude < 0.0001:
#         return (0, 0, 0, 0)
#     # Normalize the vector part
#     x /= magnitude
#     y /= magnitude
#     z /= magnitude
#     # Calculate the angle of rotation
#     angle = 2 * math.acos(q[3])
#     # Convert the angle and vector to an axis-angle
#     return (x, y, z, angle)

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_axis_angle(w, x, y, z):
    angle = 2 * math.acos(w)
    s = math.sqrt(1 - w * w)
    if s < 0.001:
        axis = np.array([x, y, z])
    else:
        axis = np.array([x / s, y / s, z / s])
    return axis, angle

def plot_axes(ax, axis, angle, loc):
    axis *= angle
    ax.quiver(loc[0], loc[1], loc[2], axis[0], axis[1], axis[2], color='r')

def main():
    with open('quaternion_to_axis_angle.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        
        data = []
        row_counter = 0
        for row in csvreader:
            row_counter += 1
            if row_counter % 5 != 0:
                continue  # Skip this row if it's not the 5th row
            axis1 = np.array([float(row[5]), float(row[6]), float(row[7])])
            angle1 = float(row[8])
            loc1 = np.array([float(row[10]), float(row[11]), float(row[12])])
            
            axis2 = np.array([float(row[13]), float(row[14]), float(row[15])])
            angle2 = float(row[16])
            loc2 = np.array([float(row[17]), float(row[18]), float(row[19])])
            
            data.append((axis1, angle1, loc1, axis2, angle2, loc2))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for axis1, angle1, loc1, axis2, angle2, loc2 in data:
        plot_axes(ax, axis1, angle1, loc1)
        plot_axes(ax, axis2, angle2, loc2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Axis-Angle Representation Visualization (Every 5th Row)')
    
    plt.show()

if __name__ == "__main__":
    main()
