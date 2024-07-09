import sys
sys.path.append('./')
import uproot
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import grand_psu_lib.utils.utils as utils
import argparse
import glob
import datetime
from matplotlib import rc
rc('font', size=13.0)


def str2bool(s):
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')





path = '/Users/ab212678/Documents/GRAND/data/auger/2024/*/'

plot_path = './position_gaa'
os.makedirs(plot_path, exist_ok=True)

list_of_files = glob.glob(path + '*MD*')

request = 'gps_lat'


def get_qty_from_list_of_files(list_of_files, request):
    res = []

    for file in list_of_files:
        print('working on file: ', file)
        try:
            trawv = uproot.open(file)['trawvoltage']
        except OSError:
            print('empty file')
        else:
            du_list = utils.get_dulist(trawv)
            for idu in du_list:
                try:
                    a, b = utils.get_column_for_given_du(trawv, request, idu)
                    qty = a.to_numpy().squeeze()
                except ValueError:
                    print("duplicate here")
                else:
                    a, b = utils.get_column_for_given_du(trawv, request, idu)
                    print(len(qty), len(b))
                    date_array = b
                    ts_list = [datetime.datetime.timestamp(d) for d in date_array ]
                    ts_array = np.array(ts_list)
                    du_array = qty.copy()*0 + idu
                    res.append([qty, ts_array, du_array])

    res = np.hstack(res)
    return res.T


ts_min = datetime.datetime(2024, 3, 1, 0 , 0, 0).timestamp()
ts_max = datetime.datetime(2024, 7, 1, 0 , 0, 0).timestamp()

res_lat = utils.get_qty_from_list_of_files(list_of_files, 'gps_lat')
res_lon = utils.get_qty_from_list_of_files(list_of_files, 'gps_long')

ind_ok = np.where(
    (res_lat[:, 0] < -10) *
    (res_lat[:, 0] > -90) *
    (res_lon[:, 0] < -10) *
    (res_lon[:, 0] > -90) *
    (res_lat[:, 1] > ts_min) *
    (res_lat[:, 1] < ts_max)
)[0]

res_lat = res_lat[ind_ok]
res_lon = res_lon[ind_ok]


du_list = np.unique(res_lat[:, 2])

for idu in du_list:
    where_du = np.where(res_lat[:, 2] == idu)[0]
    res_lat_du = res_lat[where_du]
    res_lon_du = res_lon[where_du]
    plt.figure(idu)
    plt.clf()
    plt.scatter(res_lon_du[:, 0], res_lat_du[:, 0], c=res_lat_du[:, 1]-res_lat_du[:, 1].min())
    plt.colorbar(label='time -t0 [sec]')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.title('DU {}'.format(int(idu)))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'gaa_du{}_v1.png'.format(idu)))


ind_ok2 = np.where(
    (res_lat[:, 0] < -34) *
    (res_lat[:, 0] > -36) *
    (res_lon[:, 0] < -10) *
    (res_lon[:, 0] > -90) *
    (res_lat[:, 1] > ts_min) *
    (res_lat[:, 1] < ts_max)
)[0]

res_lat = res_lat[ind_ok2]
res_lon = res_lon[ind_ok2]


du_list = np.unique(res_lat[:, 2])

for idu in du_list:
    where_du = np.where(res_lat[:, 2] == idu)[0]
    res_lat_du = res_lat[where_du]
    res_lon_du = res_lon[where_du]
    plt.figure(idu+3000)
    plt.clf()
    plt.scatter(res_lon_du[:, 0], res_lat_du[:, 0], c=res_lat_du[:, 1]-res_lat_du[:, 1].min())
    plt.colorbar(label='time -t0 [sec]')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.title('DU {}'.format(int(idu)))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'gaa_du{}_v2.png'.format(idu)))





aera_pos = np.loadtxt('../dc2_code/gaa_positions.txt', skiprows=1, usecols=[0, 2, 3, 4, 5, 6])



# gaa_pos_from_du = np.loadtxt('../dc2_code/gaa_position_lonlat.txt', skiprows=1)
# gaa_pos_from_du_v2 = np.loadtxt('../dc2_code/gaa_position_lonlat_april2024.txt', skiprows=1)

plt.figure(2, figsize=(15, 8))
plt.clf()
# plt.plot(gaa_pos_from_du[:, 1], gaa_pos_from_du[:, 2], 'r.', label='From DU march2024')
# plt.plot(gaa_pos_from_du_v2[:, 1], gaa_pos_from_du_v2[:, 2], 'g.', label='From DU april2024')

plt.plot(aera_pos[:, 2], aera_pos[:, 1], 'k.', label='from AERA file')
for pos in aera_pos:
    plt.text(pos[2], pos[1], '{}'.format(int(pos[0])))


for idu in du_list:
    where_du = np.where(res_lat[:, 2] == idu)[0]
    res_lat_du = res_lat[where_du]
    res_lon_du = res_lon[where_du]
 
    plt.scatter(res_lon_du[:, 0], res_lat_du[:, 0], c=res_lat_du[:, 1]-res_lat_du[:, 1].min())
  
  
plt.legend()
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.tight_layout()
plt.colorbar(label='time -t0 [sec]')
plt.savefig(os.path.join(plot_path, 'gaa_general_aera.png'))
