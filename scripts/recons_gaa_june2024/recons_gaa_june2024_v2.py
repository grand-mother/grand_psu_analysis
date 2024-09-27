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
import datetime
from PWF_reconstruction import recons_PWF as pwf
from numpy.linalg import LinAlgError
#import recons_PWF as pwf
import scipy
from grid_shape_lib.utils import diff_spec as tale
# rc('font', **{'family':'serif','serif':['Palatino']})
# rc('text', usetex = True)
rc('font', size = 16.0)

D2R = np.pi / 180
R2D = 180 / np.pi

path_1 = ''

gaa_pos = np.loadtxt(os.path.join('/Users/ab212678/Documents/GRAND/Codes/dc2_code_deprecated', './gaa_positions.txt'), skiprows=1, usecols=[0, 2, 3, 4, 5, 6])

plt.figure(1)
plt.clf()
plt.plot(gaa_pos[:, 4]/1000, gaa_pos[:, 3]/1000, 'k.')
for pos in gaa_pos:
    plt.text(pos[4]/1000, pos[3]/1000, '{}'.format(int(pos[0])))

plt.xlabel('East [km]')
plt.ylabel('South [km]')
plt.tight_layout()

gaa_pos_from_du = np.loadtxt(os.path.join('/Users/ab212678/Documents/GRAND/Codes/dc2_code_deprecated','gaa_position_lonlat.txt'), skiprows=1)

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



fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')

x59 = gaa_pos[gaa_pos[:, 0] == 59, 3][0]
y59 = -gaa_pos[gaa_pos[:, 0] == 59, 4][0]

def do_recons_given_list(list_ndu):
    res = []
    for file in list_ndu:

        evnum = file.split('_')[-1].split('.')[0]

        arr = np.load(file)
        
        N = []
        W = []
        z = []

        for id_du in arr[:, 0]:
            W.append(-gaa_pos[gaa_pos[:, 0] == id_du, 4][0])
            N.append(gaa_pos[gaa_pos[:, 0] == id_du, 3][0])
            z.append(gaa_pos[gaa_pos[:, 0] == id_du, 5][0])

            # if id_du == 59:
            #     x59 = N[-1]
            #     y59 = W[-1]

        x_ants = np.vstack([N, W, z]).T

        x_ants[:, 0] -= x59
        x_ants[:, 1] -= y59

        t_antsx = arr[:, 5] * 1e-9
        t_antsy = arr[:, 6] * 1e-9
        print(t_antsx)

        if (t_antsx-t_antsy).std() < 5e-4:

            try:
                #theta_x, phi_x = pwf.PWF_(x_ants, t_antsx, verbose=False, cr=1.000139, c=pwf.c_light)
                theta_x, phi_x = pwf.PWF_semianalytical(x_ants, t_antsx, n=1.000139, c=pwf.c_light)

                #pwf.PWF_(x_ants, t_antsx, verbose=False, cr=1.000139, c=pwf.c_light)
                
            except ValueError as ve:
                theta_x, phi_x = (-1, -1)
            try:
                #theta_y, phi_y = pwf.PWF_simon(x_ants, t_antsy, verbose=False, cr=1.000139, c=pwf.c_light)
                theta_y, phi_y = pwf.PWF_semianalytical(x_ants, t_antsy, n=1.000139, c=pwf.c_light)
            except ValueError as ve:
                theta_y, phi_y = (-1, -1)
            print(theta_x, phi_x)

            try:
                cov_x = pwf.Covariance_schurcomplement(theta_x, phi_x, x_ants, 5e-9, c=pwf.c_light, n=1.000139)
            except LinAlgError as lae:
                cov_x = np.array([[1, 0], [0, 1]])

            try:
                cov_y = pwf.Covariance_schurcomplement(theta_y, phi_y, x_ants, 5e-9, c=pwf.c_light, n=1.000139)
            except LinAlgError as lae:
                cov_y = np.array([[1, 0], [0, 1]])
#            cov_x = pwf.Cov_Matrix(theta_x, phi_x, x_ants, cr=1.000139, c=pwf.c_light, sigma=5e-9)
            #cov_y = pwf.Cov_Matrix(theta_y, phi_y, x_ants, cr=1.000139, c=pwf.c_light, sigma=5e-9)

            zen_x = theta_x * pwf.R2D
            az_x = phi_x * pwf.R2D
            sig_zen_x = np.sqrt(cov_x[0, 0]) * pwf.R2D
            sig_az_x = np.sqrt(cov_x[1, 1]) * pwf.R2D

            zen_y = theta_y * pwf.R2D
            az_y = phi_y * pwf.R2D
            sig_zen_y = np.sqrt(cov_y[0, 0]) * pwf.R2D
            sig_az_y = np.sqrt(cov_y[1, 1]) * pwf.R2D


            #print('ev {} x, zen = {} +- {}, az = {} +- {}'.format(file.split('_')[-1][:-4], zen_x, sig_zen_x, az_x, sig_az_x))
            #print('ev {} y, zen = {} +- {}, az = {} +- {}'.format(file.split('_')[-1][:-4], zen_y, sig_zen_y, az_y, sig_az_y))
            #print('-----')

        else:

            zen_x = zen_y = az_x = az_y = sig_zen_x = sig_zen_y = sig_az_x = sig_az_y =-1

        res.append([int(evnum), int(arr[0, 4]), zen_x, zen_y, az_x, az_y, sig_zen_x, sig_zen_y, sig_az_x, sig_az_y])
    res= np.array(res)

    return res

            # if zen < 89:
            #     ax.errorbar(az*pwf.D2R, zen, xerr=2* sig_az *pwf.D2R, yerr = 2*sig_zen, capsize=3, fmt="o", ms=1, c="darkred", label='5DUs', zorder=18 )




#### Plot May 1st data




site = 'gaa'
file_list = glob.glob('/Users/ab212678/Documents/GRAND/data/auger/2024/04/*CD*.root')
output_path = '/Users/ab212678/Documents/GRAND/data/study_gaa_recons_june24_v5_with_traces/'

plot_path_main = '/Users/ab212678/Documents/GRAND/data/study_gaa_recons_june24_v5_with_traces/'

list_dir = os.listdir(output_path)
if '.DS_Store' in list_dir:
    list_dir.remove('.DS_Store')

list_dir = np.sort(list_dir)

R3 = []
R4 = []
R5 = []
R6 = []
R7 = []

for ldir in list_dir:

    output_path_ = os.path.join(output_path, ldir)
    plot_path = os.path.join(output_path_, 'recons_plots')
    os.makedirs(plot_path, exist_ok=True)

    list_7du = np.sort(glob.glob(output_path_ + '/recons_data/7du/recons*'))
    list_6du = np.sort(glob.glob(output_path_ + '/recons_data/6du/recons*'))
    list_5du = np.sort(glob.glob(output_path_ + '/recons_data/5du/recons*'))
    list_4du = np.sort(glob.glob(output_path_ + '/recons_data/4du/recons*'))
    list_3du = np.sort(glob.glob(output_path_ + '/recons_data/3du/recons*'))



    res5 = do_recons_given_list(list_5du)
    print('5 done')
    res4 = do_recons_given_list(list_4du)
    print('4 done')
    res3 = do_recons_given_list(list_3du)
    print('3 done')

    res6 = do_recons_given_list(list_6du)
    res7 = do_recons_given_list(list_7du)

    

    fig = plt.figure(59)
    fig.clf()
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')
    if len(res3>0):
        R3.append(res3)
        ax.scatter(res3[:, 4]*pwf.D2R, res3[:, 2], c="seagreen", s=3, label='3DUs' )
    if len(res4>0):
        R4.append(res4)
        ax.scatter(res4[:, 4]*pwf.D2R, res4[:, 2], c="darkblue", s=3, label='4DUs' )
    if len(res5>0):
        R5.append(res5)
        ax.scatter(res5[:, 4]*pwf.D2R, res5[:, 2], c="red", s=3, label='5DUs' )
    if len(res6>0):
        R6.append(res6)
        ax.scatter(res6[:, 4]*pwf.D2R, res6[:, 2], c="c", s=3, label='6DUs' )
    if len(res7>0):
        R7.append(res7)
        ax.scatter(res7[:, 4]*pwf.D2R, res7[:, 2], c="m", s=10, label='7DUs' )
    plt.title(ldir, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_path, 'polar_recons_{}.png'.format(ldir)))


R3 = np.vstack(R3)
R4 = np.vstack(R4)
R5 = np.vstack(R5)
R6 = np.vstack(R6)
R7 = np.vstack(R7)

fig = plt.figure(60)
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
if len(R3>0):
    ax.scatter(R3[:, 4]*pwf.D2R, R3[:, 2], c="seagreen", s=3, label='3DUs' )
if len(R4>0):
    ax.scatter(R4[:, 4]*pwf.D2R, R4[:, 2], c="darkblue", s=3, label='4DUs' )
if len(R5>0):
    ax.scatter(R5[:, 4]*pwf.D2R, R5[:, 2], c="red", s=3, label='5DUs' )
if len(R6>0):
    ax.scatter(R6[:, 4]*pwf.D2R, R6[:, 2], c="c", s=3, label='6DUs' )
if len(R7>0):
    ax.scatter(R7[:, 4]*pwf.D2R, R7[:, 2], c="m", s=10, label='7DUs' )
    
############################################################
# investigate gaa_20240426_130829_RUN003002_CD_phys
############################################################
output_path = '/Users/ab212678/Documents/GRAND/data/study_gaa_recons_june24_v5_with_traces/'
ldir = 'gaa_20240426_130829_RUN003002_CD_phys'
output_path_ = os.path.join(output_path, ldir)
list_alldu = np.sort(glob.glob(output_path_ + '/recons_data/*/recons*'))
res = do_recons_given_list(list_alldu)
res[res[:, 4]>200,4] -=360

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[:, 4]*pwf.D2R, res[:, 2], c=res[:, 1]-res[:, 1].min(), s=3, cmap='jet' )
plt.colorbar(sc, label='time -t0 [sec]')
fig.tight_layout()
date_initial = datetime.datetime.fromtimestamp(int(res[:, 1].min()), tz=tz_GMT)

id2000 = np.where(((res[:, 1]-res[:,1].min()) < 1900)*((res[:, 1]-res[:,1].min())>1850))[0]


date_0 = datetime.datetime(2024, 4, 26, 13, 50, 0, 0, tz_GMT)
ts0 = datetime.datetime.timestamp(date_0)


fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[id2000, 2], c=res[id2000, 1]-ts0, s=3, cmap='jet' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(datetime.datetime.fromtimestamp(int(ts0), tz=tz_GMT)))
fig.tight_layout()





 

############################################################
# investigate  gaa_20240413_070637_RUN003002_CD_phys

############################################################
output_path = '/Users/ab212678/Documents/GRAND/data/study_gaa_recons_june24_v5_with_traces/'
ldir = 'gaa_20240413_070637_RUN003002_CD_phys'
output_path_ = os.path.join(output_path, ldir)
plot_path = os.path.join(output_path_, 'recons_plots')
list_alldu = np.sort(glob.glob(output_path_ + '/recons_data/*/recons*'))
res = do_recons_given_list(list_alldu)
res[res[:, 4]>200,4] -=360

plt.figure()
plt.scatter(res[:,4], res[:, 2], c=res[:, 1]-res[:, 1].min(), s=4, cmap='Paired')
plt.colorbar(label='time -t0 [sec]')
plt.xlabel('Azimuth [deg]')
plt.ylabel('Zenith [deg]')
plt.tight_layout()


fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[:, 4]*pwf.D2R, res[ :, 2], c=res[:, 1]-res[:, 1].min(), s=3, cmap='Paired' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(datetime.datetime.fromtimestamp(int(res[:, 1].min()), tz=tz_GMT)))
fig.tight_layout()
fig.savefig(os.path.join(plot_path, 'April13th_all.png'))




id2000 = np.where(((res[:, 1]-res[:,1].min()) < 10400)*((res[:, 1]-res[:,1].min())>9000))[0]



date_0 = datetime.datetime(2024, 4, 13, 10, 5, 00, 0, tz_GMT)
ts0 = datetime.datetime.timestamp(date_0)

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[ id2000, 2], c=res[id2000, 1]-ts0, s=3, cmap='Paired' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(date_0))
fig.tight_layout()
fig.savefig(os.path.join(plot_path, 'April13th_10h05.png'))




id2000 = np.where(((res[:, 1]-res[:,1].min()) < 10400)*((res[:, 1]-res[:,1].min())>9000))[0]


date_0 = datetime.datetime(2024, 4, 13, 10, 8, 50,0, tz_GMT)
ts0 = datetime.datetime.timestamp(date_0)

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[ id2000, 2], c=res[id2000, 1]-ts0, s=3, cmap='Paired' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(date_0))
fig.tight_layout()
fig.savefig(os.path.join(plot_path, 'April13th_10h08_50.png'))



id2000 = np.where(((res[:, 1]-res[:,1].min()) < 11600)*((res[:, 1]-res[:,1].min())>10500))[0]

date_0 = datetime.datetime(2024, 4, 13, 10, 28, 0,0, tz_GMT)
ts0 = datetime.datetime.timestamp(date_0)

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[ id2000, 2], c=res[id2000, 1]-ts0, s=3, cmap='Paired' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(date_0))
fig.tight_layout()
fig.savefig(os.path.join(plot_path, 'April13th_10h28.png'))




id2000 = np.where(((res[:, 1]-res[:,1].min()) < 11600)*((res[:, 1]-res[:,1].min())>10500))[0]

date_0 = datetime.datetime(2024, 4, 13, 10, 23, 0,0, tz_GMT)
ts0 = datetime.datetime.timestamp(date_0)

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[ id2000, 2], c=res[id2000, 1]-ts0, s=3, cmap='Paired' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(date_0))
fig.tight_layout()
fig.savefig(os.path.join(plot_path, 'April13th_10h23.png'))





id2000 = np.where(((res[:, 1]-res[:,1].min()) < 11600)*((res[:, 1]-res[:,1].min())>11000))[0]

plt.figure()
plt.scatter(res[id2000, 4], res[ id2000, 2], c=res[id2000, 1]-res[id2000[0], 1], s=5)
plt.colorbar(label='time -t0 [sec]')
plt.title('UTC t0 = {}'.format(datetime.datetime.fromtimestamp(int(res[id2000[0], 1]), tz=tz_GMT)))
plt.xlabel('Azimuth [deg]')
plt.ylabel('Zenith [deg]')
plt.tight_layout()
 



fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[ id2000, 2], c=res[id2000, 1]-res[[id2000[0]], 1], s=3, label='3DUs' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(datetime.datetime.fromtimestamp(int(res[id2000[0], 1]), tz=tz_GMT)))
fig.tight_layout()


plt.figure()
plt.scatter(res[:,4], res[:, 2], c=res[:, 1]-res[:, 1].min(), s=4, cmap='Paired')
plt.colorbar(label='time -t0 [sec]')
plt.xlim((98, 102))
plt.ylim((65, 90))
plt.xlabel('Azimuth [deg]')
plt.ylabel('Zenith [deg]')
plt.tight_layout()


###########################################################
# investigate gaa_20240415_142228_RUN003002_CD_phys


############################################################
output_path = '/Users/ab212678/Documents/GRAND/data/study_gaa_recons_june24_v5_with_traces/'
ldir = 'gaa_20240415_142228_RUN003002_CD_phys'
output_path_ = os.path.join(output_path, ldir)
list_alldu = np.sort(glob.glob(output_path_ + '/recons_data/*/recons*'))
res = do_recons_given_list(list_alldu)
res[res[:, 4]>200,4] -=360

plt.figure()
plt.scatter(res[:,4], res[:, 2], c=res[:, 1]-res[:, 1].min(), s=4, cmap='Paired')
plt.colorbar(label='time -t0 [sec]')
plt.xlabel('Azimuth [deg]')
plt.ylabel('Zenith [deg]')
plt.tight_layout()


id2000 = np.where(((res[:, 1]-res[:,1].min()) < 8000)*((res[:, 1]-res[:,1].min())>7500))[0]

plt.figure()
plt.scatter(res[id2000, 4], res[ id2000, 2], c=res[id2000, 1]-res[:, 1].min(), s=5)
plt.colorbar(label='time -t0 [sec]')
plt.xlabel('Azimuth [deg]')
plt.ylabel('Zenith [deg]')
plt.tight_layout()
 
plt.figure()
plt.scatter(res[:,4], res[:, 2], c=res[:, 1]-res[:, 1].min(), s=4, cmap='Paired')
plt.colorbar(label='time -t0 [sec]')
plt.xlim((98, 102))
plt.ylim((65, 90))
plt.xlabel('Azimuth [deg]')
plt.ylabel('Zenith [deg]')
plt.tight_layout()





###########################################################
# investigate gaa_20240420_171234_RUN003002_CD_phys


############################################################
output_path = '/Users/ab212678/Documents/GRAND/data/study_gaa_recons_june24_v5_with_traces/'
ldir = 'gaa_20240420_171234_RUN003002_CD_phys'
output_path_ = os.path.join(output_path, ldir)
plot_path = os.path.join(output_path_, 'recons_plots')
list_alldu = np.sort(glob.glob(output_path_ + '/recons_data/*/recons*'))
res = do_recons_given_list(list_alldu)
res[res[:, 4]>200,4] -=360

plt.figure()
plt.scatter(res[:,4], res[:, 2], c=res[:, 1]-res[:, 1].min(), s=4, cmap='Paired')
plt.colorbar(label='time -t0 [sec]')
plt.xlabel('Azimuth [deg]')
plt.ylabel('Zenith [deg]')
plt.tight_layout()


id2000 = np.where(((res[:, 1]-res[:,1].min()) < 11640)*((res[:, 1]-res[:,1].min())>11620))[0]

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[ id2000, 2], c=res[id2000, 1]-res[[id2000[0]], 1], s=3, label='3DUs' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(datetime.datetime.fromtimestamp(int(res[id2000[0], 1]), tz=tz_GMT)))
fig.tight_layout()
## This one has no conterpart on flightradar







id2000 = np.where(((res[:, 1]-res[:,1].min()) < 70000)*((res[:, 1]-res[:,1].min())>69590))[0]
fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[ id2000, 2], c=res[id2000, 1]-res[[id2000[0]], 1], s=3, label='3DUs' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(datetime.datetime.fromtimestamp(int(res[id2000[0], 1]), tz=tz_GMT)))
fig.tight_layout()













date_0 = datetime.datetime(2024, 4, 21, 3, 19, 0,0, tz_GMT)
ts0 = datetime.datetime.timestamp(date_0)

## Flight 3807
id2000 = np.where(((res[:, 1]-res[:,1].min()) < 38000)*((res[:, 1]-res[:,1].min())>35000))[0]
fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[ id2000, 2], c=res[id2000, 1]-ts0, s=3, label='3DUs' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(datetime.datetime.fromtimestamp(ts0, tz=tz_GMT)))
fig.tight_layout()
fig.savefig(os.path.join(plot_path, 'april21st_3h19.png.png'))




###########################################################
# investigate  gaa_20240425_191722_RUN003002_CD_phys


############################################################
output_path = '/Users/ab212678/Documents/GRAND/data/study_gaa_recons_june24_v5_with_traces/'
ldir = 'gaa_20240425_191722_RUN003002_CD_phys'
output_path_ = os.path.join(output_path, ldir)
list_alldu = np.sort(glob.glob(output_path_ + '/recons_data/*/recons*'))
res = do_recons_given_list(list_alldu)
res[res[:, 4]>200,4] -=360

plt.figure()
plt.scatter(res[:,4], res[:, 2], c=res[:, 1]-res[:, 1].min(), s=4, cmap='Paired')
plt.colorbar(label='time -t0 [sec]')
plt.xlabel('Azimuth [deg]')
plt.ylabel('Zenith [deg]')
plt.tight_layout()


id2000 = np.where(((res[:, 1]-res[:,1].min()) < 1300)*((res[:, 1]-res[:,1].min())>1000))[0]


date_0 = datetime.datetime(2024, 4, 25, 20, 2, 0,0, tz_GMT)
ts0 = datetime.datetime.timestamp(date_0)

fig = plt.figure()
fig.clf()
ax = fig.add_subplot(projection='polar')
ax.set_theta_zero_location('N')
sc = ax.scatter(res[id2000, 4]*pwf.D2R, res[ id2000, 2], c=res[id2000, 1]-ts0, s=3, label='3DUs' )
plt.colorbar(sc, label='time -t0 [sec]')
ax.set_title('UTC t0 = {}'.format(datetime.datetime.fromtimestamp(ts0, tz=tz_GMT)))
fig.tight_layout()


plt.figure()
plt.scatter(res[id2000, 4], res[ id2000, 2], c=res[id2000, 1]-res[:, 1].min(), s=5, cmap='Paired')
plt.colorbar(label='time -t0 [sec]')
plt.xlabel('Azimuth [deg]')
plt.ylabel('Zenith [deg]')
plt.tight_layout()
 

















if False:

    list_7du = np.sort(glob.glob(output_path + '*2405*/recons_data/7du/recons*'))
    list_6du = np.sort(glob.glob(output_path + '*2405*/recons_data/6du/recons*'))
    list_5du = np.sort(glob.glob(output_path + '*2405*/recons_data/5du/recons*'))
    list_4du = np.sort(glob.glob(output_path + '*2405*/recons_data/4du/recons*'))
    list_3du = np.sort(glob.glob(output_path + '*2405*/recons_data/3du/recons*'))


    res5 = do_recons_given_list(list_5du)
    print('5 done')
    res4 = do_recons_given_list(list_4du)
    print('4 done')
    res3 = do_recons_given_list(list_3du)
    print('3 done')

    res6 = do_recons_given_list(list_6du)
    res7 = do_recons_given_list(list_7du)

    ts0 = res3[:, 1].min()

    plt.figure(111)
    plt.clf()

    if len(list_5du)>0:

        plt.scatter(res5[:, 4], res5[:, 2], marker='x', c=(res5[:, 1]-ts0)/60/60)
        plt.scatter(res5[:, 5], res5[:, 3], marker='v', c=(res5[:, 1]-ts0)/60/60)


    plt.scatter(res4[:, 4], res4[:, 2], marker='o', c=(res4[:, 1]-ts0)/60/60)
    plt.scatter(res4[:, 5], res4[:, 3], marker='.', c=(res4[:, 1]-ts0)/60/60)

    plt.colorbar()




    plt.figure(112)
    plt.clf()
    #plt.scatter(res3[:, 4], res3[:, 2], marker='x', c=(res3[:, 1]-ts0)/60/60)
    plt.scatter(res3[:, 5], res3[:, 3], marker='.', c=np.log10(res3[:, 1]-ts0))

    id3 = np.where(   (res3[:, 5]<75)*(0< res3[:, 5])*(24<res3[:, 3])*(res3[:, 3]<30) )



    plt.colorbar()

    mask3 = np.where((res3[:, 2]>0)*(res3[:, 3]>0))[0]





    # fig = plt.figure(56)
    # fig.clf()
    # ax = fig.add_subplot(projection='polar')
    # ax.set_theta_zero_location('N')
    # ax.hist2d(res3[mask3, 4]*pwf.D2R, res3[mask3, 2], bins=100, norm='log')


    # fig = plt.figure(57)
    # fig.clf()
    # ax = fig.add_subplot(projection='polar')
    # ax.set_theta_zero_location('N')
    # ax.hist2d(res4[:, 4]*pwf.D2R, res4[:, 2], bins=100, norm='log')


    # fig = plt.figure(60)
    # fig.clf()
    # ax = fig.add_subplot(projection='polar')
    # ax.set_theta_zero_location('N')
    # ax.hist2d(res6[:, 4]*pwf.D2R, res6[:, 2], bins=100, norm='log')

    # fig = plt.figure(61)
    # fig.clf()
    # ax = fig.add_subplot(projection='polar')
    # ax.set_theta_zero_location('N')
    # ax.hist2d(res7[:, 4]*pwf.D2R, res7[:, 2], bins=100, norm='log')


    # if len(list_5du) > 0:
    #     fig = plt.figure(58)
    #     fig.clf()
    #     ax = fig.add_subplot(projection='polar')
    #     ax.set_theta_zero_location('N')
    #     ax.hist2d(res5[:, 4]*pwf.D2R, res5[:, 2], bins=30)


    fig = plt.figure(59)
    fig.clf()
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')
    if len(res3>0):
        ax.scatter(res3[:, 4]*pwf.D2R, res3[:, 2], c="seagreen", s=3, label='3DUs' )
    if len(res4>0):
        ax.scatter(res4[:, 4]*pwf.D2R, res4[:, 2], c="darkblue", s=3, label='4DUs' )
    if len(res5>0):
        ax.scatter(res5[:, 4]*pwf.D2R, res5[:, 2], c="red", s=3, label='5DUs' )
    if len(res6>0):
        ax.scatter(res6[:, 4]*pwf.D2R, res6[:, 2], c="c", s=3, label='6DUs' )
    if len(res7>0):
        ax.scatter(res7[:, 4]*pwf.D2R, res7[:, 2], c="m", s=10, label='7DUs' )





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