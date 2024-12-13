import os
import utils_recons_gp80 as ur80
import glob


''' Scripts that extract the coincidence and does the recons for the beacon tests of nov 1st and nov 2nd to study the time offset
'''


data_path = '/Users/ab212678/Documents/GRAND/data/gp13/2024/11/'

output_dir = '/Users/ab212678/Documents/GRAND/data/gp13/study_timedelays_GP13_11DU*7CD_gps_status_flag/'

file_list = glob.glob(data_path + 'GP13_20241101*11DU*Mover*Point[2-7]*.root')

for fname in file_list:

    coincs, geo_info_xyz = ur80.find_coincs_in_udfile_withvalid_du34(fname, 7, do_plots=True)
    if len(coincs) > 0:
        ur80.get_data_from_coincs_beacon(coincs, geo_info_xyz, fname, output_dir=output_dir, do_recons=True)
