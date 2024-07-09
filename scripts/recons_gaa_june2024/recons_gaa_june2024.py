import sys
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
import grand_psu_lib.utils.utils as utils
import grand_psu_lib.utils.filtering as filt
# better plots
from matplotlib import rc
import uproot
import os
import json
from scipy.signal import hilbert
import glob
import recons_PWF as pwf
import scipy
from grid_shape_lib.utils import diff_spec as tale
# rc('font', **{'family':'serif','serif':['Palatino']})
# rc('text', usetex = True)
rc('font', size = 16.0)

D2R = np.pi / 180
R2D = 180 / np.pi


gaa_pos = np.loadtxt('./gaa_positions.txt', skiprows=1, usecols=[0, 2, 3, 4, 5, 6])

plt.figure(1)
plt.clf()
plt.plot(gaa_pos[:, 4]/1000, gaa_pos[:, 3]/1000, 'k.')
for pos in gaa_pos:
    plt.text(pos[4]/1000, pos[3]/1000, '{}'.format(int(pos[0])))

plt.xlabel('East [km]')
plt.ylabel('South [km]')
plt.tight_layout()

gaa_pos_from_du = np.loadtxt('gaa_position_lonlat.txt', skiprows=1)

plt.figure(2)
plt.clf()
plt.plot(gaa_pos_from_du[:, 1], gaa_pos_from_du[:, 2], 'r.', label='From DU')
plt.plot(gaa_pos[:, 2], gaa_pos[:, 1], 'k.', label='from AERA file')
for pos in gaa_pos:
    plt.text(pos[2], pos[1], '{}'.format(int(pos[0])))
plt.legend()
plt.xlabel('Longitude [deg]')
plt.xlabel('Latitude [deg]')
plt.tight_layout()
plt.savefig('gaa_position_comp1.png')



plt.figure(3)
plt.clf()
plt.plot(gaa_pos_from_du[:, 1]-gaa_pos_from_du[2, 1], gaa_pos_from_du[:, 2]-gaa_pos_from_du[2, 2], 'r.', label='From DU')
plt.plot(gaa_pos[:, 2]-gaa_pos[2, 2], gaa_pos[:, 1]-gaa_pos[2, 1], 'k.', label='from AERA file')
for pos in gaa_pos:
    plt.text(pos[2]-gaa_pos[2, 2], pos[1]-gaa_pos[2, 1], '{}'.format(int(pos[0])))
plt.legend()
plt.xlabel('delta Longitude [deg]')
plt.xlabel('delta Latitude [deg]')
plt.tight_layout()
plt.savefig('gaa_position_comp2.png')




center_gaa_SN = np.mean(gaa_pos[:, 3])
center_gaa_EW = np.mean(gaa_pos[:, 4])

gaa_pos[:, 3] -= center_gaa_SN
gaa_pos[:, 4] -= center_gaa_EW



plt.figure(3)
plt.clf()
plt.plot(gaa_pos[:, 4]/1000, gaa_pos[:, 3]/1000, 'k.')
for pos in gaa_pos:
    plt.text(pos[4]/1000, pos[3]/1000, '{}'.format(int(pos[0])))

plt.xlabel('Easting [km]')
plt.ylabel('Northing [km]')
plt.tight_layout()





data_5du_path = '/Users/ab212678/Documents/GRAND/Codes/grand_psu_analysis/gaa_5du_gaa_20240427_224428_RUN003002_CD_phys/'

data_4du_path = '/Users/ab212678/Documents/GRAND/Codes/grand_psu_analysis/good_du4/'
data_4du_path = '/Users/ab212678/Documents/GRAND/Codes/grand_psu_analysis/gaa_4du_gaa_20240427_224428_RUN003002_CD_phys/'
list_5du = np.sort(glob.glob(data_5du_path + '/*.npy'))
list_4du = np.sort(glob.glob(data_4du_path + '/*.npy'))



fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')


for file in list_5du:

    arr = np.load(file)
    N = []
    W = []
    z = []

    for id_du in arr[:, 0]:
        W.append(-gaa_pos[gaa_pos[:, 0] == id_du, 4][0])
        N.append(gaa_pos[gaa_pos[:, 0] == id_du, 3][0])
        z.append(gaa_pos[gaa_pos[:, 0] == id_du, 5][0]-1500)

        if id_du == 59:
            x59 = N[-1]
            y59 = W[-1]

    x_ants = np.vstack([N, W, z]).T
    x_ants[:, 0] -= x59
    x_ants[:, 1] -= y59

    print(x_ants.shape)
    t_ants = arr[:, 1]*1e-9

    theta, phi = pwf.PWF_simon(x_ants, t_ants, verbose=False, cr=1.000139, c=pwf.c_light)

    cov = pwf.Cov_Matrix(theta, phi, x_ants, cr=1.00013, c=pwf.c_light, sigma=5e-9)

    zen = theta * pwf.R2D
    az = phi * pwf.R2D

    sig_zen = np.sqrt(cov[0, 0]) * pwf.R2D
    sig_az = np.sqrt(cov[1, 1]) * pwf.R2D

    print('ev {}, zen = {} +- {}, az = {} +- {}'.format(file.split('_')[-1][:-4], zen, sig_zen, az, sig_az))
    if zen < 89:
        ax.errorbar(az*pwf.D2R, zen, xerr=2* sig_az *pwf.D2R, yerr = 2*sig_zen, capsize=3, fmt="o", ms=1, c="darkred", label='5DUs', zorder=18 )


    plt.figure()
    plt.clf()
    plt.plot(gaa_pos[:, 4], gaa_pos[:, 3], 'ko', alpha=0.3)
    for pos in gaa_pos:
        plt.text(pos[4], pos[3], '{}'.format(int(pos[0])))

    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')

    plt.scatter(-x_ants[:, 1], x_ants[:, 0], c=t_ants*1e6, zorder=6)
    plt.colorbar(label='arrival time [us]')
    plt.tight_layout()
    plt.savefig('Recons_5du_{}.png'.format(file.split('_')[-1][:-4]))


ax.set_ylim(0, 100)
ax.set_title('Reconstruction of events with 5dus')
fig.savefig('gaa_april2024_5du.png')




## Load lphne results

coord_antennas = np.loadtxt('./gaa_tables/coord_antennas.txt')
pos_lphne = coord_antennas[0:5, :]

plt.figure(45)
plt.clf()
plt.plot(x_ants[:, 0], x_ants[:, 1], 'ko', ms =10, label='abl')
plt.plot(pos_lphne[:, 1], pos_lphne[:, 2], 'r.', ms=5, label='lpnhe')
plt.xlabel('Northing [m]')
plt.ylabel('Easting [m]')
plt.legend()
plt.tight_layout()
plt.savefig('gaa_position_abl_lpnhe.png')


























# fig = plt.figure()
# ax = fig.add_subplot(projection='polar')
# ax.set_theta_zero_location('N')


# for file in list_4du:
#     # file = list_5du[0]

#     arr = np.load(file)
#     N = []
#     W = []
#     z = []

#     for id_du in arr[:, 0]:
#         W.append(-gaa_pos[gaa_pos[:, 0]==id_du, 4][0])
#         N.append(gaa_pos[gaa_pos[:, 0]==id_du, 3][0])
#         z.append(gaa_pos[gaa_pos[:, 0]==id_du, 5][0])


#     x_ants = np.vstack([N, W, z]).T
#     t_ants = arr[:, 1]*1e-9

#     theta, phi = pwf.PWF_simon(x_ants, t_ants, verbose=False, cr=1.00013, c=pwf.c_light)

#     cov = pwf.Cov_Matrix(theta, phi, x_ants, cr=1.00013, c=pwf.c_light, sigma=5e-9)

#     zen = theta * pwf.R2D
#     az = phi * pwf.R2D

#     sig_zen = np.sqrt(cov[0, 0]) * pwf.R2D
#     sig_az = np.sqrt(cov[1, 1]) * pwf.R2D

#     print('ev {}, zen = {} +- {}, az = {} +- {}'.format(file.split('_')[-1][:-4], zen, sig_zen, az, sig_az))
#     if zen < 89:
#         #ax.scatter(az*pwf.D2R, zen, c="seagreen", s=3, label='4DUs' )
#         ax.errorbar(az*pwf.D2R, zen, xerr= 2 *sig_az *pwf.D2R, yerr = 2*sig_zen, capsize=3, fmt="o", c="seagreen", ms=1, label='4DUs' )

# ax.set_ylim(0, 100)
# ax.set_title('Reconstruction of events with 4dus')
# plt.savefig('gaa_april2024_4du.png')

# plt.figure(4)
# plt.clf()
# plt.plot(-gaa_pos[:, 4], gaa_pos[:, 3], 'ko', alpha=0.3)
# plt.xlabel('westing [m]')
# plt.scatter(x_ants[:, 1], x_ants[:, 0], c=t_ants, zorder=6)
# plt.tight_layout()

# plt.figure(5)
# plt.clf()
# plt.plot(gaa_pos[:, 4], gaa_pos[:, 3], 'ko', alpha=0.3)
# plt.xlabel('Easting [m]')
# plt.scatter(-x_ants[:, 1], x_ants[:, 0], c=t_ants, zorder=6)
# plt.tight_layout()