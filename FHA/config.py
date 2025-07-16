# config.py

# General
data_type = '1'  # '1' for Polhemus, '2' for Vicon
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/1_90deg_ydata_oneway.csv'
path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/4_90deg_y_with_xxdeg_z_rotationdata.csv'
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/2_90deg_y_with_translationdata.csv'

# Only needed for Vicon
marker1_j1 = 'LASI'
marker2_j1 = 'RASI'
marker3_j1 = 'SACR'
marker1_j2 = 'LKNE'
marker2_j2 = 'RKNE'
marker3_j2 = 'LANK'

##### FHA calculation ######

# method_type = 'all_FHA'
# method_type = 'incremental_time'
# method_type = 'step_angle'
method_type = 'incremental_angle'

cut1 = 0  # start index for data cut
cut2 = 1000
step = 15  # ignored if method_type == 'all_FHA'
nn = 20