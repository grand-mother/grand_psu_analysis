
import numpy as np
import matplotlib.pyplot as plt
# better plots
from matplotlib import rc
import os
import json
from grand_psu_lib.utils import utils_dc2_abl as udc2
from scipy.signal import hilbert
import argparse

rc('font', size=11.0)


def str2bool(s):
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    ap = argparse.ArgumentParser(description='Extract information for ZhaireS Root files for PWF/SWF DC@ reconstruction')
    ap.add_argument(
        "--sim_path",
        type=str,
        required=True,
        help="directy of the root file e.g .../sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_0000"
    )
    ap.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="output path for the .npy and .json files"
    )
    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    simdir = args.sim_path
    output_path = args.output_path

    d1 = udc2.dd(simdir)
    event_list = d1.event_list

    plot_path = os.path.join(output_path, 'plots_events')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    for i in range(len(event_list)):

        print('doing event {}'.format(event_list[i][0]))
        print('loading event')
        d1.get_event(*event_list[i])
        d1.load_event(*event_list[i], ef_l1_mode=False)

        print('event loaded')
        ev_stuff = []
        ev_stuff_v2 = []
        ev_stuff.append(d1.ev_du_list)
        ev_stuff.append(d1.ev_du_position)

        ev_stuff_v2.append(d1.ev_du_list)
        ev_stuff_v2.append(d1.ev_du_position)

        tmax_ef_l1 = []
        max_ef_l1 = []

        tmax_tadc_l1 = []
        max_tadc_l1 = []

        tmax_ef_v2 = []
        Amax_ef_v2 = []
        tmax_adc_v2 = []
        Amax_adc_v2 = []



        for du_id in d1.ev_du_list:
            idu = np.where(d1.ev_dus_indices == du_id)[0][0]
            ### Efield l1
            ef_l1_Hx = np.abs(hilbert(d1.ev_trace_efield_l1[idu, 0]))
            ef_l1_Hy = np.abs(hilbert(d1.ev_trace_efield_l1[idu, 1]))
            ef_l1_Hz = np.abs(hilbert(d1.ev_trace_efield_l1[idu, 2]))

            norme_ef_xyz = np.sqrt(d1.ev_trace_efield_l1[idu, 0]**2 + d1.ev_trace_efield_l1[idu, 1]**2 + d1.ev_trace_efield_l1[idu, 2]**2)
            norm_ef_H_xyz = np.abs(hilbert(norme_ef_xyz))

            norme_ef_xy = np.sqrt(d1.ev_trace_efield_l1[idu, 0]**2 + d1.ev_trace_efield_l1[idu, 1]**2)
            norm_ef_H_xy = np.abs(hilbert(norme_ef_xy))

            norme_H_ef_xyz = np.sqrt(np.abs(hilbert(d1.ev_trace_efield_l1[idu, 0]))**2 + np.abs(hilbert(d1.ev_trace_efield_l1[idu, 1]))**2+np.abs(hilbert(d1.ev_trace_efield_l1[idu, 2]))**2) 
            norme_H_ef_xy = np.sqrt(np.abs(hilbert(d1.ev_trace_efield_l1[idu, 0]))**2 + np.abs(hilbert(d1.ev_trace_efield_l1[idu, 1]))**2)

            imax_norm_ef_H_xyz = np.argmax(norm_ef_H_xyz)
            imax_norm_ef_H_xy = np.argmax(norm_ef_H_xy)
            imax_norme_H_ef_xyz = np.argmax(norme_H_ef_xyz)
            imax_norme_H_ef_xy = np.argmax(norme_H_ef_xy)

            tmax_norm_ef_H_xyz = d1.ev_trace_efield_l1_time[imax_norm_ef_H_xyz] + d1.ev_t0_efield_l1[idu]
            tmax_norm_ef_H_xy = d1.ev_trace_efield_l1_time[imax_norm_ef_H_xy] + d1.ev_t0_efield_l1[idu]
            tmax_norme_H_ef_xyz = d1.ev_trace_efield_l1_time[imax_norme_H_ef_xyz] + d1.ev_t0_efield_l1[idu]
            tmax_norme_H_ef_xy = d1.ev_trace_efield_l1_time[imax_norme_H_ef_xy] + d1.ev_t0_efield_l1[idu]

            Emax_norm_ef_H_xyz = norm_ef_H_xyz[imax_norm_ef_H_xyz]
            Emax_norm_ef_H_xy = norm_ef_H_xy[imax_norm_ef_H_xy]
            Emax_norme_H_ef_xyz = norme_H_ef_xyz[imax_norme_H_ef_xyz]
            Emax_norme_H_ef_xy = norme_H_ef_xy[imax_norme_H_ef_xy]

            imax_ef_l1_Hx = np.argmax(ef_l1_Hx)
            imax_ef_l1_Hy = np.argmax(ef_l1_Hy)
            imax_ef_l1_Hz = np.argmax(ef_l1_Hz)

            tmax_ef_l1_Hx = d1.ev_trace_efield_l1_time[imax_ef_l1_Hx] + d1.ev_t0_efield_l1[idu]
            tmax_ef_l1_Hy = d1.ev_trace_efield_l1_time[imax_ef_l1_Hy] + d1.ev_t0_efield_l1[idu]
            tmax_ef_l1_Hz = d1.ev_trace_efield_l1_time[imax_ef_l1_Hz] + d1.ev_t0_efield_l1[idu]

            Emax_ef_l1_Hx = ef_l1_Hx[imax_ef_l1_Hx]
            Emax_ef_l1_Hy = ef_l1_Hy[imax_ef_l1_Hy]
            Emax_ef_l1_Hz = ef_l1_Hz[imax_ef_l1_Hz]

            tmax_ef_l1.append([tmax_ef_l1_Hx, tmax_ef_l1_Hy, tmax_ef_l1_Hz])
            max_ef_l1.append([Emax_ef_l1_Hx, Emax_ef_l1_Hy, Emax_ef_l1_Hz])

            tmax_ef_v2.append([tmax_norm_ef_H_xyz, tmax_norm_ef_H_xy, tmax_norme_H_ef_xyz, tmax_norme_H_ef_xy])
            Amax_ef_v2.append([Emax_norm_ef_H_xyz, Emax_norm_ef_H_xy, Emax_norme_H_ef_xyz, Emax_norme_H_ef_xy])

            ### tadc L1
            tadc_l1_Hx = np.abs(hilbert(d1.ev_trace_ADC_l1[idu, 0]))
            tadc_l1_Hy = np.abs(hilbert(d1.ev_trace_ADC_l1[idu, 1]))
            tadc_l1_Hz = np.abs(hilbert(d1.ev_trace_ADC_l1[idu, 2]))

            imax_tadc_l1_Hx = np.argmax(tadc_l1_Hx)
            imax_tadc_l1_Hy = np.argmax(tadc_l1_Hy)
            imax_tadc_l1_Hz = np.argmax(tadc_l1_Hz)

            tmax_tadc_l1_Hx = d1.ev_trace_ADC_l1_time[imax_tadc_l1_Hx] + d1.ev_t0_adc_l1[idu]
            tmax_tadc_l1_Hy = d1.ev_trace_ADC_l1_time[imax_tadc_l1_Hy] + d1.ev_t0_adc_l1[idu]
            tmax_tadc_l1_Hz = d1.ev_trace_ADC_l1_time[imax_tadc_l1_Hz] + d1.ev_t0_adc_l1[idu]

            Emax_tadc_l1_Hx = tadc_l1_Hx[imax_tadc_l1_Hx]
            Emax_tadc_l1_Hy = tadc_l1_Hy[imax_tadc_l1_Hy]
            Emax_tadc_l1_Hz = tadc_l1_Hz[imax_tadc_l1_Hz]

            tmax_tadc_l1.append([tmax_tadc_l1_Hx, tmax_tadc_l1_Hy, tmax_tadc_l1_Hz])
            max_tadc_l1.append([Emax_tadc_l1_Hx, Emax_tadc_l1_Hy, Emax_tadc_l1_Hz])

            norme_adc_xyz = np.sqrt(d1.ev_trace_ADC_l1[idu, 0]**2 + d1.ev_trace_ADC_l1[idu, 1]**2 + d1.ev_trace_ADC_l1[idu, 2]**2)
            norm_adc_H_xyz = np.abs(hilbert(norme_adc_xyz))

            norme_adc_xy = np.sqrt(d1.ev_trace_ADC_l1[idu, 0]**2 + d1.ev_trace_ADC_l1[idu, 1]**2)
            norm_adc_H_xy = np.abs(hilbert(norme_adc_xy))

            norme_H_adc_xyz = np.sqrt(np.abs(hilbert(d1.ev_trace_ADC_l1[idu, 0]))**2 + np.abs(hilbert(d1.ev_trace_ADC_l1[idu, 1]))**2+np.abs(hilbert(d1.ev_trace_ADC_l1[idu, 2]))**2) 
            norme_H_adc_xy = np.sqrt(np.abs(hilbert(d1.ev_trace_ADC_l1[idu, 0]))**2 + np.abs(hilbert(d1.ev_trace_ADC_l1[idu, 1]))**2)

            imax_norm_adc_H_xyz = np.argmax(norm_adc_H_xyz)
            imax_norm_adc_H_xy = np.argmax(norm_adc_H_xy)
            imax_norme_H_adc_xyz = np.argmax(norme_H_adc_xyz)
            imax_norme_H_adc_xy = np.argmax(norme_H_adc_xy)

            tmax_norm_adc_H_xyz = d1.ev_trace_ADC_l1_time[imax_norm_adc_H_xyz] + d1.ev_t0_adc_l1[idu]
            tmax_norm_adc_H_xy = d1.ev_trace_ADC_l1_time[imax_norm_adc_H_xy] + d1.ev_t0_adc_l1[idu]
            tmax_norme_H_adc_xyz = d1.ev_trace_ADC_l1_time[imax_norme_H_adc_xyz] + d1.ev_t0_adc_l1[idu]
            tmax_norme_H_adc_xy = d1.ev_trace_ADC_l1_time[imax_norme_H_adc_xy] + d1.ev_t0_adc_l1[idu]

            Emax_norm_adc_H_xyz = norm_adc_H_xyz[imax_norm_adc_H_xyz]
            Emax_norm_adc_H_xy = norm_adc_H_xy[imax_norm_adc_H_xy]
            Emax_norme_H_adc_xyz = norme_H_adc_xyz[imax_norme_H_adc_xyz]
            Emax_norme_H_adc_xy = norme_H_adc_xy[imax_norme_H_adc_xy]

            tmax_adc_v2.append([tmax_norm_adc_H_xyz, tmax_norm_adc_H_xy, tmax_norme_H_adc_xyz, tmax_norme_H_adc_xy])
            Amax_adc_v2.append([Emax_norm_adc_H_xyz, Emax_norm_adc_H_xy, Emax_norme_H_adc_xyz, Emax_norme_H_adc_xy])


        ev_stuff.append(np.array(tmax_ef_l1))
        ev_stuff.append(np.array(max_ef_l1))
        ev_stuff.append(np.array(tmax_tadc_l1))
        ev_stuff.append(np.array(max_tadc_l1))

        ev_stuff_v2.append(np.array(tmax_ef_v2))
        ev_stuff_v2.append(np.array(Amax_ef_v2))
        ev_stuff_v2.append(np.array(tmax_adc_v2))
        ev_stuff_v2.append(np.array(Amax_adc_v2))


        ev_stuff_arr = np.hstack([
            np.expand_dims(ev_stuff[0], 1),  # ev_du_list
            ev_stuff[1],  # ev_du_position
            ev_stuff[2],  # tmax_ef_l1
            ev_stuff[3],  # max_ef_l1
            ev_stuff[4],  # tmax_tadc_l1
            ev_stuff[5]   # max_tadc_l1
        ])
        ev_stuff_arr_v2 = np.hstack([
            np.expand_dims(ev_stuff_v2[0], 1),  # ev_du_list
            ev_stuff_v2[1],  # ev_du_position
            ev_stuff_v2[2], # tmaxs ef
            ev_stuff_v2[3],  #Emaxs ef
            ev_stuff_v2[4], # tmaxs adc
            ev_stuff_v2[5], # Amax adc
        ])


        json_file = os.path.join(output_path, '{}.json'.format(d1.event_params["event_number"]))
        with open(json_file, 'w') as f:
            json.dump(d1.event_params, f)

        arr_file = os.path.join(output_path, '{}.npy'.format(d1.event_params["event_number"]))
        np.save(arr_file, ev_stuff_arr)

        arr_file_v2 = os.path.join(output_path, '{}_v2.npy'.format(d1.event_params["event_number"]))
        np.save(arr_file_v2, ev_stuff_arr_v2)
        # sometimes do some plots
        if np.random.rand(1) < 0.0005:
            fig, ax = plt.subplots(1, 1)
            sc = ax.scatter(-ev_stuff[1][:, 1], ev_stuff[1][:, 0], c=ev_stuff[3].max(axis=1))
            ax.set_ylabel('Northing [m]')
            ax.set_xlabel('Easting [m]')
            fig.colorbar(sc, label='Emax [uV/m]')
            ax.set_title('ev n. {} Emax '.format(d1.event_params["event_number"]))
            fig.tight_layout()
            fig.savefig(os.path.join(plot_path, 'event{}_Emax_.png'.format(d1.event_params["event_number"])))

            fig, ax = plt.subplots(1, 1)
            sc = ax.scatter(-ev_stuff[1][:, 1], ev_stuff[1][:, 0], c=ev_stuff[2].max(axis=1))
            ax.set_ylabel('Northing [m]')
            ax.set_xlabel('Easting [m]')
            fig.colorbar(sc, label='tmax [ns]')
            ax.set_title('ev n. {} tmax '.format(d1.event_params["event_number"]))
            fig.tight_layout()
            fig.savefig(os.path.join(plot_path, 'event{}_tmax_.png'.format(d1.event_params["event_number"])))
