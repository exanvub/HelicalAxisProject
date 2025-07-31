# config.py

# General
data_type = '1'  # '1' for Polhemus
# data_type = '2'  # Vicon
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/1_90deg_ydata_oneway.csv'
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/4_90deg_y_with_xxdeg_z_rotationdata.csv'
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/2_90deg_y_with_translationdata.csv'
path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/3_90deg_y_with_xxdeg_x_rotationdata.csv'
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Vicon_test_data/servo_knee_90degX.c3d'
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Vicon_test_data/servo_knee_90degX_and_20Y.c3d'



# # Only needed for Vicon
# marker1_j1 = 'LASI'
# marker2_j1 = 'RASI'
# marker3_j1 = 'SACR'
# marker1_j2 = 'LKNE'
# marker2_j2 = 'RKNE'
# marker3_j2 = 'LANK'

marker1_j1 = 'lateral_epicondyl'
marker2_j1 = 'medial_epicondyl'
marker3_j1 = 'caput'
marker1_j2 = 'ankle'
marker2_j2 = 'lateral_condyl'
marker3_j2 = 'medial_condyl'

##### FHA calculation ######

# method_type = 'step_angle'
method_type = 'incremental_angle'

cut1 = 100  # start index for data cut
cut2 = 1500
step = 10 # ignored if method_type == 'all_FHA'
nn = 1