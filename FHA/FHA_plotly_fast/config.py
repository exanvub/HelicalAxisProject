# config.py — starting values for the dashboard.
#
# Everything here can also be changed in the dashboard itself. Edit this file to change
# what the dashboard opens with, then press "Reload config.py" (or refresh the page).

# ----- Recording --------------------------------------------------------------------
data_type = '1'    # '1' Polhemus CSV, '2' Vicon C3D, '3' built-in simulator (known true axis)
# data_type = '2'  # Vicon
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/1_90deg_ydata_oneway.csv'
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/4_90deg_y_with_xxdeg_z_rotationdata.csv'
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/2_90deg_y_with_translationdata.csv'
path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/3_90deg_y_with_xxdeg_x_rotationdata.csv'
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Vicon_test_data/servo_knee_90degX.c3d'
# path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Vicon_test_data/servo_knee_90degX_and_20Y.c3d'

# Length unit of Polhemus positions. The CSV does not say; 'cm' matches the bench rig.
polhemus_unit = 'cm'

# ----- Simulator only (data_type = '3') ---------------------------------------------
# A hinge plus an optional second rotation, with the true axis known at every frame.
# 'none' pure hinge | 'proportional' steady second rotation | 'late' phased in smoothly
# 'pulsed' second rotation delivered in bursts
sim_profile = 'proportional'
sim_noise = 'polhemus'       # 'none', 'polhemus' or 'vicon' (calibrated on the test recordings)
sim_out_and_back = False

# ----- Vicon only -------------------------------------------------------------------
# The first three markers of each segment define its frame (origin midway between
# markers 2 and 3). Segment 1 is the reference, segment 2 the moving one.
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

# Further markers on the same segments, e.g. ['femur_1', 'femur_2']. They do not change
# what the frames mean, only how precisely they are estimated (least-squares fit only).
extra_markers_j1 = []
extra_markers_j2 = []

# 'least_squares': fit one rigid body to all markers of a segment (recommended).
# 'three_marker' : build each frame directly from its three markers, as before.
vicon_frames = 'least_squares'

# ----- Smoothing --------------------------------------------------------------------
# Low-pass filter applied to the motion before the axes are computed: cut-off in Hz
# (-3 dB, no time shift, close to the usual zero-lag 4th-order Butterworth). 0 = off.
# The dashboard's slider offers 20 15 10 8 6 5 4 3 2 1.5 1 0.7 0.5 0.3 0.2 Hz; another
# value here opens on the nearest of those.
# A filter cannot tell noise from movement: it removes whatever changes faster than the
# cut-off. The dashboard shows how much that is, next to the frame-to-frame jitter of
# the recording, and draws the unsmoothed axes underneath for comparison.
smoothing_hz = 0

# ----- FHA calculation --------------------------------------------------------------
# 'incremental_angle' moving window, one axis per frame (miFHA)
# 'step_angle'        consecutive, non-overlapping windows (fiFHA)
# 'reference_pose'    axis of the displacement since the first selected frame
# 'incremental_time'  fixed lag of `step` frames
# 'all_FHA'           consecutive frames
# method_type = 'step_angle'
method_type = 'incremental_angle'

cut1 = 100   # first frame to analyse
cut2 = 1500  # last frame to analyse
step = 10    # window: degrees for the angle methods, frames for 'incremental_time'; ignored for 'all_FHA'
nn = 1       # no longer used: the dashboard draws at most `max_axes` axes, spaced evenly in rotation

# 'net' : a window closes when the NET rotation between its two ends reaches `step` (recommended)
# 'path': when the summed frame-to-frame angles reach `step` — the behaviour up to ISB 2025
window_rule = 'net'

# 'endpoints': classical FHA between the two end poses of each window (recommended)
# 'fit'      : least squares over all frames in the window; only helps against frame-to-frame jitter
estimator = 'endpoints'

# ----- Build-up of the second rotation ----------------------------------------------
# The build-up view adds up how far the moving segment has turned away from a primary
# axis. Measured against a recording's OWN average axis that reads too low, because the
# average axis already contains the second rotation. Measured against the axis of a
# recording of the primary rotation alone (same reference sensor mounting), it shows
# the second rotation adding up.
# 'reference': use the average axis of `reference_path`   'average': use this recording's own
build_up_axis = 'reference'
reference_path = '/Users/nicolas/Github/exanvub/HelicalAxisProject/test_data/Polhemus_test_data/1_90deg_ydata.csv'

# ----- Display ----------------------------------------------------------------------
display_frame = 'auto'   # 'auto', 'lab' or 'body' (frame of the reference sensor)
plane_offset = 0.0       # distance of the cross-section plane from the pivot, along the average axis
color_by = 'time'        # 'time' or 'angle'
max_axes = 300           # how many axes to draw
theme = 'light'          # 'light' or 'dark'
