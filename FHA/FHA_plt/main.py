import polhemus_import as pi
import vicon_import as vi
import FHA_visualizer_plt as fhav
import config

def main():
    isPolhemus = False
    methods = ['all_FHA', 'incremental_time', 'step_angle', 'incremental_angle']

    if config.data_type == '1':
        q1, q2, loc1, loc2, t = pi.read_data(config.path)
        R1, T1, R2, T2 = pi.quaternions_to_matrices(q1, q2, loc1, loc2)
        isPolhemus = True

    elif config.data_type == '2':
        c3d, marker_data, t = vi.read_data(config.path)
        
        labels = c3d['parameters']['POINT']['LABELS']['value']
        for m in [config.marker1_j1, config.marker2_j1, config.marker3_j1, config.marker1_j2, config.marker2_j2, config.marker3_j2]:
            if m not in labels:
                raise ValueError(f"Marker {m} not found in c3d file.")

        R1, T1, R2, T2, loc1, loc2 = vi.build_reference_frames(
            marker_data,
            config.marker1_j1, config.marker2_j1, config.marker3_j1,
            config.marker1_j2, config.marker2_j2, config.marker3_j2
        )

    if config.method_type not in methods:
        raise ValueError(f"FHA method '{config.method_type}' not in supported methods {methods}")

    step = config.step if config.method_type != 'all_FHA' else 0

    hax, ang, svec, d, translation_1_list, translation_2_list, time_diff, time_incr, ang_incr, all_angles, ind_incr, ind_step, t = fhav.generate_FHA(
        config.method_type, t, int(config.cut1), int(config.cut2), int(step), int(config.nn),
        R1, R2, T1, T2, loc1, loc2
    )

    fhav.visualize_FHA(
        isPolhemus, T1, T2, config.method_type, hax, svec, d, ind_incr, ind_step, int(config.nn),
        translation_1_list, translation_2_list, t, time_incr, ang_incr, int(step), all_angles, time_diff
    )

if __name__ == "__main__":
    main()
