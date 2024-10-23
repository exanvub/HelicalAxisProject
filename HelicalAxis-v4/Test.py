import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.transform import Rotation as R

data = pd.read_csv('/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/1_90deg_ydata.csv')


# Extract relevant data
time = data['Time'].values
quat1 = data[['x1', 'y1', 'z1', 'w1']].values
quat2 = data[['x2', 'y2', 'z2', 'w2']].values
loc1 = data[['loc1_x', 'loc1_y', 'loc1_z']].values
loc2 = data[['loc2_x', 'loc2_y', 'loc2_z']].values

def calculate_angular_difference(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r = r1 * r2.inv()
    return r.as_rotvec()

ha_list = []

for i in range(len(time)):
    q1 = quat1[i]
    q2 = quat2[i]
    ha = calculate_angular_difference(q1, q2)
    # fha_norm = fha / np.linalg.norm(fha)
    ha_list.append(ha)
    # fha_norm_list.append(fha_norm)

print(ha_list)

# plot rotation vectors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for i in range(len(ha_list)):
    ha = ha_list[i]
    ax.quiver(0, 0, 0, ha[0], ha[1], ha[2])

plt.show()




