import math
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/IMU_test_data/3xflex-ext.csv'
data = pd.read_csv(csv_file_path)

# Check if the required columns are in the dataframe
required_columns = ['x_diff', 'y_diff', 'z_diff']
if all(column in data.columns for column in required_columns):
    # Plot the data
    plt.figure(figsize=(10, 6))

    plt.plot(data['x_diff'], label='x')
    plt.plot(data['y_diff'], label='y')
    plt.plot(data['z_diff'], label='z')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Diff X, Y, and Z Over Time')
    plt.legend()
    plt.grid(True)

    plt.show()
else:
    print(f"CSV file is missing one or more required columns: {required_columns}")
