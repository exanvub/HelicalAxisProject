# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 10:33:07 2025

@author: Matteo Iurato
"""

import polhemus_import as pi
import polhemus_import as pi
import vicon_import as vi
import FHA_visualizer as fhav

def main():
    isPolhemus = False
    
    methods = ['all_FHA', 'incremental_time', 'step_angle', 'incremental_angle']
    
    data_type = input('Insert 1 for Polhemus data or 2 for Vicon data, then press Enter: ')
    while(data_type != '1' and data_type!= '2'):
        data_type = input('Please insert 1 for Polhemus data or 2 for Vicon data, then press Enter: ')
    
    if(data_type == '1'):
        path = input('Please provide the path of your .csv file and then press Enter: ')
        q1, q2, loc1, loc2, t = pi.read_data(path)
        R1, T1, R2, T2 = pi.quaternions_to_matrices(q1, q2, loc1, loc2)
        isPolhemus = True
    
    elif(data_type == '2'):
        #Acquire c3d file path and read data
        path = input('Please provide the path of your .c3d file and then press Enter: ')
        c3d, marker_data, t = vi.read_data(path)
        
        #Acquire marker labels and build reference systems
        marker1_j1 = input('Please provide the name of the first marker to build reference frame of joint 1, then press Enter: ')
        while(marker1_j1 not in c3d['parameters']['POINT']['LABELS']['value']):
            marker1_j1 = input('First marker for joint 1 not found. Please provide a valid label for the marker and then press Enter: ')
        
        marker2_j1 = input('Please provide the name of the second marker to build reference frame of joint 1, then press Enter: ')
        while(marker2_j1 not in c3d['parameters']['POINT']['LABELS']['value']):
            marker2_j1 = input('Second marker for joint 1 not found. Please provide a valid label for the marker and then press Enter: ')
        
        marker3_j1 = input('Please provide the name of the third marker to build reference frame of joint 1, then press Enter: ')
        while(marker3_j1 not in c3d['parameters']['POINT']['LABELS']['value']):
            marker3_j1 = input('Third marker for joint 1 not found. Please provide a valid label for the marker and then press Enter: ')
        
        marker1_j2 = input('Please provide the name of the first marker to build reference frame of joint 2, then press Enter: ')
        while(marker1_j2 not in c3d['parameters']['POINT']['LABELS']['value']):
            marker1_j2 = input('First marker for joint 2 not found. Please provide a valid label for the marker and then press Enter: ')
        
        marker2_j2 = input('Please provide the name of the second marker to build reference frame of joint 2, then press Enter: ')
        while(marker2_j2 not in c3d['parameters']['POINT']['LABELS']['value']):
            marker2_j2 = input('Second marker for joint 2 not found. Please provide a valid label for the marker and then press Enter: ')
        
        marker3_j2 = input('Please provide the name of the third marker to build reference frame of joint 2, then press Enter: ')
        while(marker3_j2 not in c3d['parameters']['POINT']['LABELS']['value']):
            marker3_j2 = input('Third marker for joint 2 not found. Please provide a valid label for the marker and then press Enter: ')
        
        R1, T1, R2, T2, loc1, loc2 = vi.build_reference_frames(marker_data, marker1_j1, marker2_j1, marker3_j1, marker1_j2, marker2_j2, marker3_j2)

    print('These are the available FHA calculation methods. Please choose one of the following: \n')
    print(methods)
    method_type = input('\n Please type the chosen method for FHA calculation, then press Enter: ')
    while(method_type not in methods):
        print('The method type is not among the available ones. Please type a valid method and then press Enter: ')
    
    cut1 = input('Please type the lower bound of the range to which you want your data to be cut and then press Enter: ')
    cut2 = input('Please type the upper bound of the range to which you want your data to be cut and then press Enter: ')
    if(method_type != 'all_FHA'):
        step = input('Please type the desired step to calculate FHA and then press Enter: ')
    else:
        step = 0
    nn = input('Please type the ratio to which you want to reduce the number of FHA for visualization, then press Enter: ')
    

    hax, ang, svec, d, translation_1_list, translation_2_list, time_diff, time_incr, ang_incr, all_angles, ind_incr, ind_step, t = fhav.generate_FHA(method_type, t, int(cut1), int(cut2), int(step), int(nn), R1, R2, T1, T2, loc1, loc2)
    fhav.visualize_FHA(isPolhemus, T1, T2, method_type, hax, svec, d, ind_incr, ind_step, int(nn), translation_1_list, translation_2_list, t, time_incr, ang_incr, int(step), all_angles, time_diff)

    
if __name__=="__main__":
    main()