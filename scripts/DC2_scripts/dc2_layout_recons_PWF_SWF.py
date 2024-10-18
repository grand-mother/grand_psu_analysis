

import numpy as np
import glob
import matplotlib.pyplot as plt

# better plots
from matplotlib import rc
from scipy import interpolate as interp
import os
from grid_shape_lib.modules import masks as masks
from grand_psu_lib.modules import layout_dc2 as ldu


# rc('font', **{'family':'serif','serif':['Palatino']})
# rc('text', usetex = True)
rc('font', size = 16.0)

D2R = np.pi / 180
R2D = 180 / np.pi


class Binning:
    def __init__(
        self,
        zen_min=30,
        zen_max=89,
        n_bin_zen=10,
        energy_min=16.4,
        energy_max=18.7,
        n_bin_energy=9
    ):
        self.zen_min = zen_min
        self.zen_max = zen_max
        self.n_bin_zen = n_bin_zen

        self.energy_min = energy_min
        self.energy_max = energy_max
        self.n_bin_energy = n_bin_energy
        self.create_bins()

    def create_bins(self):
        self.zen_bin_edges = np.linspace(self.zen_min, self.zen_max, self.n_bin_zen+1)
        self.zen_bin_centers = 0.5 * (self.zen_bin_edges[1:] + self.zen_bin_edges[:-1])

        self.energy_bin_edges = np.linspace(self.energy_min, self.energy_max, self.n_bin_energy+1)
        self.energy_bin_centers = 0.5 * (self.energy_bin_edges[1:] + self.energy_bin_edges[:-1])

        self.delta_zen = self.zen_bin_edges[1:] - self.zen_bin_edges[:-1]
        self.delta_energy = 10**(self.energy_bin_edges[1:]) - 10**(self.energy_bin_edges[:-1])





def load_all_antennas():
    # this event contains all the 289 antennas
    ev_id = 85
    res_file = os.path.join(data_dir_zhaires_nj, 'data_files/{}.npy'.format(ev_id))

    arr = np.load(res_file)
    du_names = arr[:, 0]
    du_pos = arr[:, 1:4]

    return du_pos, du_names



n_trig_thres = 4


data_pwf_swf = '/Users/ab212678/Documents/GRAND/sims/DC2/DC2Training/PWF_SWF_data/'

data_dir_zhaires_nj = '/Users/ab212678/Documents/GRAND/sims/DC2/DC2Training/PWFdata_3june24/Zhaires_NJ'



output_dir = '/Users/ab212678/Documents/GRAND/sims/DC2/study_DC2_pwf_swf/'



gp13_id = [138, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288]

du_pos_all, du_names_all = load_all_antennas()


mask1000 = (
    (np.round(du_pos_all[:, 0]/(np.sqrt(3)/2), 0) % 1000 == 0) *
    (np.round(du_pos_all[:, 1], 0) % 500 == 0) *
    (np.round(du_pos_all[:, 1], 0) != 0)
)

gp1000_names = list(du_names_all[mask1000])
gp1000_names.remove(77)
gp1000_names.remove(73)
gp1000_names.remove(106)
gp1000_names.remove(203)
gp1000_names.remove(170)
gp1000_names.remove(199)



# Faisage du layout infill only
d = np.sqrt(du_pos_all[:, 0]**2+du_pos_all[:, 1]**2) 

ids_infill = list(np.where(d < 1950)[0])

for id in gp13_id[1:]:
    ids_infill.remove(id)


ids_infill500 = [201, 170, 138, 106, 75, 215, 187, 154, 124, 91, 63, 203, 172, 140, 108, 77, 156, 126, 213, 185, 152, 122, 89, 61, 199, 168, 136, 104, 73, 150, 120]



lay_289_60_10_ncall50 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='lay289_60_10_13000_ncall50',
    output_dir=output_dir,
    threshold=60, n_trig_thres=10,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=True, ncall=50
)
lay_289_60_10_ncall50.make_plots()



lay_289_60_10_ncall75 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='lay289_60_10_13000_ncall75',
    output_dir=output_dir,
    threshold=60, n_trig_thres=10,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=True, ncall=75
)
lay_289_60_10_ncall75.make_plots()


lay_289_60_10_ncall100 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='lay289_60_10_13000_ncall100',
    output_dir=output_dir,
    threshold=60, n_trig_thres=10,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=True, ncall=100
)
lay_289_60_10_ncall100.make_plots()


lay_289_60_10_ncall150 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='lay289_60_10_13000_ncall150',
    output_dir=output_dir,
    threshold=60, n_trig_thres=10,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=True, ncall=150
)
lay_289_60_10_ncall150.make_plots()



lay_289_60_10_ncall100_lownoise = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='lay289_60_10_13000_ncall100_lownoise',
    output_dir=output_dir,
    threshold=60, n_trig_thres=10,
    do_noise_timing=True,
    sigma_timing=1e-10, do_swf=True, ncall=100
)
lay_289_60_10_ncall100_lownoise.make_plots()






lay_289_60_10_ncall200 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='lay289_60_10_13000_ncall200',
    output_dir=output_dir,
    threshold=60, n_trig_thres=10,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=True, ncall=200
)
lay_289_60_10_ncall200.make_plots()



lay_289_60_10_ncall200_lownoise = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='lay289_60_10_13000_ncall200_lownoise',
    output_dir=output_dir,
    threshold=60, n_trig_thres=10,
    do_noise_timing=True,
    sigma_timing=1e-10, do_swf=True, ncall=200
)
lay_289_60_10_ncall200_lownoise.make_plots()




# lay_289_50_10_ncall400 = ldu.Layout_dc2(
#     du_pos_all, du_names_all,
#     data_pwf_swf, layout_name='lay289_50_10_1000_ncall400',
#     output_dir=output_dir,
#     threshold=50, n_trig_thres=10,
#     do_noise_timing=True,
#     sigma_timing=5e-9, do_swf=True, ncall=400
# )
# lay_289_50_10_ncall400.make_plots()






def thetaphi_to_k(theta, phi):
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K =  - np.array([st*cp, st*sp, ct])
    return K

def k_to_theta_phi(k):
    theta = np.arccos(-k[2])
    cp = - k[0]/np.sin(theta)
    sp = - k[1]/np.sin(theta)
    phi = np.arctan2(sp, cp)
    # if phi < 0:
    #     phi += 2 * np.pi
    return theta, phi

#######################################################
##### plot proxy laws for swf initial guess ###########
#######################################################

lay1 = lay_289_60_10_ncall100
evt = lay1.event_res_tab

binned_xmax_z = ldu.bin_array(evt[:, 2], evt[:, 18], lay1.zen_bins_limits)
d2d_max = np.sqrt((evt[:, 17] - 0*evt[:, 5])**2 + (evt[:, 16] - 0*evt[:, 4])**2)




    

plt.figure(98)
plt.clf()
plt.plot(lay1.event_res_tab[:, 2], d2d_max/1000, 'k.', ms=1, alpha=0.2)    
plt.plot(lay1.zen_bins_centers, lay1.binned_dmax[0]/1000 )
plt.xlim(25, 90)
plt.xlabel('Zenith [deg]')
plt.ylabel('2d distance to Xmax [km]')
plt.yscale('log')
plt.tight_layout()
plt.savefig('./proxy_d2_dxmax.png')


plt.figure(99)
plt.clf()
plt.plot(lay1.event_res_tab[:, 2], lay1.event_res_tab[:, 18]/1000, 'k.', ms=1, alpha=0.2)    
plt.plot(lay1.zen_bins_centers, lay1.binned_xmax_z[0]/1000 )
plt.xlabel('Zenith [deg]')
plt.ylabel('Xmax altitude [km]')
plt.xlim(25, 90)
plt.tight_layout()
plt.savefig('./proxy_xmax_z.png')









core_alt = 1264


# event_res_tab[k, 24] = swf_fit_v2[0]
# event_res_tab[k, 25] = swf_fit_v2[1]
# event_res_tab[k, 26] = swf_fit_v2[2]


# event_res_tab[k, 4] = np.float32(event_params['shower_core_x'])
# event_res_tab[k, 5] = np.float32(event_params['shower_core_y'])

lay1 = lay_289_60_10_ncall50
lay2 = lay_289_60_10_ncall75
lay3 = lay_289_60_10_ncall100
lay4 = lay_289_60_10_ncall150
lay5 = lay_289_60_10_ncall200

lay6 = lay_289_60_10_ncall100_lownoise
lay7 = lay_289_60_10_ncall200_lownoise

evt1 = lay1.event_res_tab
evt2 = lay2.event_res_tab
evt3 = lay3.event_res_tab
evt4 = lay4.event_res_tab
evt5 = lay5.event_res_tab


idd1 = np.where(evt1[:, 19]!= -1)[0]

plt.figure(34567)
plt.clf()
plt.scatter(evt1[idd1, 16]/1000, evt1[idd1, 35]/1000, s=3, label='ncall=50')
plt.scatter(evt2[idd1, 16]/1000, evt2[idd1, 35]/1000, s=3, label='ncall=75')
plt.scatter(evt3[idd1, 16]/1000, evt3[idd1, 35]/1000, s=3, label='ncall=100')
plt.scatter(evt4[idd1, 16]/1000, evt4[idd1, 35]/1000, s=3, label='ncall=150')
plt.scatter(evt5[idd1, 16]/1000, evt5[idd1, 35]/1000, s=3, label='ncall=200')


plt.figure(34567)
plt.clf()
plt.scatter(evt1[idd1, 18]/1000, evt1[idd1, 37]/1000, s=3, label='ncall=50')
plt.scatter(evt2[idd1, 18]/1000, evt2[idd1, 37]/1000, s=3, label='ncall=75')
plt.scatter(evt3[idd1, 18]/1000, evt3[idd1, 37]/1000, s=3, label='ncall=100')
plt.scatter(evt4[idd1, 18]/1000, evt4[idd1, 37]/1000, s=3, label='ncall=150')
plt.scatter(evt5[idd1, 18]/1000, evt5[idd1, 37]/1000, s=3, label='ncall=200')


plt.figure(34560)
plt.clf()
plt.hist(evt1[idd1, 37]/1000, bins=40, range=(0, 20), label='ncall=50')
plt.hist(evt2[idd1, 37]/1000, bins=40, range=(0, 20), label='ncall=50')
plt.hist(evt3[idd1, 37]/1000, bins=40, range=(0, 20), label='ncall=50')
plt.hist(evt4[idd1, 37]/1000, bins=40, range=(0, 20), label='ncall=50')
plt.hist(evt5[idd1, 37]/1000, bins=40, range=(0, 20), label='ncall=50')


















idd1 = np.where(evt1[:, 19]!= -1)[0]
plt.figure(500)
plt.clf()

plt.plot(evt1[idd1, 16]/1000, evt1[idd1, 16]/1000, 'k-')

plt.scatter(evt1[idd1, 16]/1000, evt1[idd1, 19]/1000, c='g', s=3, label='swf 4d')
plt.scatter(evt1[idd1, 16]/1000, evt1[idd1, 24]/1000, c='c', s=3, label='swf 3d xmax')
plt.scatter(evt1[idd1, 16]/1000, evt1[idd1, 35]/1000, c='m', s=3, label='swf 3d xmax/2')

plt.title('ncall = 50')
plt.legend()
plt.xlabel('xmax_x [km]')
plt.ylabel('Reconstructed x_eff [km]')
plt.tight_layout()
##############################



idd2 = np.where(evt2[:, 19]!= -1)[0]
plt.figure(501)
plt.clf()

plt.plot(evt2[idd2, 16]/1000, evt2[idd2, 16]/1000, 'k-')

plt.scatter(evt2[idd2, 16]/1000, evt2[idd2, 19]/1000, c='g', s=3, label='swf 4d')
plt.scatter(evt2[idd2, 16]/1000, evt2[idd2, 24]/1000, c='c', s=3, label='swf 3d xmax')
plt.scatter(evt2[idd2, 16]/1000, evt2[idd2, 35]/1000, c='m', s=3, label='swf 3d xmax/2')
plt.title('ncall = 75')
plt.legend()
plt.xlabel('xmax_x [km]')
plt.tight_layout()


idd3 = np.where(evt3[:, 19]!= -1)[0]
plt.figure(502)
plt.clf()

plt.plot(evt3[idd3, 16]/1000, evt3[idd3, 16]/1000, 'k-')

plt.scatter(evt3[idd3, 16]/1000, evt3[idd3, 19]/1000, c='g', s=3, label='swf 4d')
plt.scatter(evt3[idd3, 16]/1000, evt3[idd3, 24]/1000, c='c', s=3, label='swf 3d xmax')
plt.scatter(evt3[idd3, 16]/1000, evt3[idd3, 35]/1000, c='m', s=3, label='swf 3d xmax/2')
plt.title('ncall = 100')
plt.legend()
plt.xlabel('xmax_x [km]')
plt.tight_layout()




idd4 = np.where(evt4[:, 19]!= -1)[0]
plt.figure(503)
plt.clf()

plt.plot(evt4[idd4, 16]/1000, evt4[idd4, 16]/1000, 'k-')

plt.scatter(evt4[idd4, 16]/1000, evt4[idd4, 19]/1000, c='g', s=3, label='swf 4d')
plt.scatter(evt4[idd4, 16]/1000, evt4[idd4, 24]/1000, c='c', s=3, label='swf 3d xmax')
plt.scatter(evt4[idd4, 16]/1000, evt4[idd4, 35]/1000, c='m', s=3, label='swf 3d xmax/2')
plt.title('ncall = 150')
plt.legend()
plt.xlabel('xmax_x [km]')
plt.tight_layout()



idd5 = np.where(evt5[:, 19]!= -1)[0]
plt.figure(504)
plt.clf()

plt.plot(evt5[idd5, 16]/1000, evt5[idd5, 16]/1000, 'k-')

plt.scatter(evt5[idd5, 16]/1000, evt5[idd5, 19]/1000, c='g', s=3, label='swf 4d')
plt.scatter(evt5[idd5, 16]/1000, evt5[idd5, 24]/1000, c='c', s=3, label='swf 3d xmax')
plt.scatter(evt5[idd5, 16]/1000, evt5[idd5, 35]/1000, c='m', s=3, label='swf 3d xmax/2')
plt.title('ncall = 200')
plt.legend()
plt.xlabel('xmax_x [km]')
plt.tight_layout()










plt.figure(4556)
plt.clf()
plt.hist(100*(lay5.deff3dv2/lay5.dmax -1), bins=40, range =[-50, 100], histtype='step')
plt.hist(100*(lay5.deff3dv1/lay5.dmax -1), bins=40, range =[-50, 100], histtype='step')

plt.hist(100*(lay4.deff3dv2/lay4.dmax -1), bins=40, range =[-50, 100], histtype='step')
plt.hist(100*(lay4.deff3dv1/lay4.dmax -1), bins=40, range =[-50, 100], histtype='step')

plt.hist(100*(lay3.deff3dv2/lay3.dmax -1), bins=20, range =[-50, 100], histtype='step')
plt.hist(100*(lay3.deff3dv1/lay3.dmax -1), bins=20, range =[-50, 100], histtype='step')

plt.hist(100*(lay2.deff3dv2/lay2.dmax -1), bins=20, range =[-50, 100], histtype='step')
plt.hist(100*(lay2.deff3dv1/lay2.dmax -1), bins=20, range =[-50, 100], histtype='step')
plt.hist(100*(lay1.deff3dv2/lay2.dmax -1), bins=20, range =[-50, 100], histtype='step')
plt.hist(100*(lay1.deff3dv1/lay2.dmax -1), bins=20, range =[-50, 100], histtype='step')





plt.figure(454)
plt.clf()
plt.hist(100*(lay7.deff3dv2/lay7.dmax -1), bins=20, range =[-50, 100], histtype='step')
plt.hist(100*(lay7.deff3dv1/lay7.dmax -1), bins=20, range =[-50, 100], histtype='step')

plt.hist(100*(lay6.deff3dv2/lay6.dmax -1), bins=20, range =[-50, 100], histtype='step')
plt.hist(100*(lay6.deff3dv1/lay6.dmax -1), bins=20, range =[-50, 100], histtype='step')




plt.figure(101)
plt.clf()
plt.hist(lay1.res_phi, bins=50, range=[-3, 3], histtype='step',  label='PWF', density=True)
plt.hist(lay2.res_phi, bins=50, range=[-3, 3], histtype='step',  label='PWF', density=True)
plt.hist(evt1[idd1, 23]-evt1[idd1, 3], bins=50, range=[-3, 3], histtype='step', label='SWF 4d', density=True)
plt.hist(evt1[idd1, 28]-evt1[idd1, 3], bins=50, range=[-3, 3], histtype='step', label='SWF 3d xmax', density=True)
plt.hist(evt1[idd1, 39]-evt1[idd1, 3], bins=50, range=[-3, 3], histtype='step', label='SWF 3d xmax/2', density=True)
plt.legend()
plt.xlabel('Azimuth residues [deg]')
plt.ylabel('# events')
plt.tight_layout()

plt.figure(102)
plt.clf()
plt.hist(lay2.res_phi, bins=50, range=[-3, 3], histtype='step',  label='PWF', density=True)
plt.hist(evt2[idd1, 23]-evt2[idd1, 3], bins=50, range=[-3, 3], histtype='step', label='SWF 4d', density=True)
plt.hist(evt2[idd1, 28]-evt2[idd1, 3], bins=50, range=[-3, 3], histtype='step', label='SWF 3d xmax', density=True)
plt.hist(evt2[idd1, 39]-evt2[idd1, 3], bins=50, range=[-3, 3], histtype='step', label='SWF 3d xmax/2', density=True)
plt.legend()
plt.xlabel('Azimuth residues [deg]')
plt.ylabel('# events')
plt.tight_layout()

plt.figure(103)
plt.clf()
plt.hist(lay3.res_phi, bins=50, range=[-3, 3], histtype='step',  label='PWF', density=True)
plt.hist(evt3[idd1, 23]-evt3[idd1, 3], bins=50, range=[-3, 3], histtype='step', label='SWF 4d', density=True)
plt.hist(evt3[idd1, 28]-evt3[idd1, 3], bins=50, range=[-3, 3], histtype='step', label='SWF 3d xmax', density=True)
plt.hist(evt3[idd1, 39]-evt3[idd1, 3], bins=50, range=[-3, 3], histtype='step', label='SWF 3d xmax/2', density=True)
plt.legend()
plt.xlabel('Azimuth residues [deg]')
plt.ylabel('# events')
plt.tight_layout()





















lay = lay_289_50_10_ncall50_ct_lown
evt = lay.event_res_tab


idd = np.where(evt[:, 19]!= -1)[0]
plt.figure(498)
plt.clf()

plt.plot(evt[idd, 16]/1000, evt[idd, 16]/1000, 'k-')
plt.scatter(evt[idd, 16]/1000, evt[idd, 19]/1000, c='g', s=3, label='swf_fit1')
plt.scatter(evt[idd, 16]/1000, evt[idd, 24]/1000, c='c', s=3, label='swf_fit2')
plt.scatter(evt[idd, 16]/1000, evt[idd, 30]/1000, c='m', s=3, label='xeff')

plt.legend()
plt.xlabel('xmax_x [km]')
plt.tight_layout()



lay = lay_289_50_10_ncall50_v2
evt = lay.event_res_tab


idd = np.where(evt[:, 19]!= -1)[0]
plt.figure(501)
plt.clf()

plt.plot(evt[idd, 16]/1000, evt[idd, 16]/1000, 'k-')
plt.scatter(evt[idd, 16]/1000, evt[idd, 19]/1000, c='g', s=3, label='swf_fit1')
plt.scatter(evt[idd, 16]/1000, evt[idd, 24]/1000, c='c', s=3, label='swf_fit2')
plt.scatter(evt[idd, 16]/1000, evt[idd, 30]/1000, c='m', s=3, label='xeff')

plt.legend()
plt.xlabel('xmax_x [km]')
plt.tight_layout()






lay = lay_289_50_10_ncall75_v2
evt = lay.event_res_tab


idd = np.where(evt[:, 19]!= -1)[0]
plt.figure(502)
plt.clf()

plt.plot(evt[idd, 16]/1000, evt[idd, 16]/1000, 'k-')
plt.scatter(evt[idd, 16]/1000, evt[idd, 19]/1000, c='g', s=3, label='swf_fit1')
plt.scatter(evt[idd, 16]/1000, evt[idd, 24]/1000, c='c', s=3, label='swf_fit2')
plt.scatter(evt[idd, 16]/1000, evt[idd, 30]/1000, c='m', s=3, label='xeff')

plt.legend()
plt.xlabel('xmax_x [km]')
plt.tight_layout()







lay1_75_5_NJ_ncall200 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='all289_NJ_75_5_all13000_ncall200',
    output_dir=output_dir,
    threshold=75, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9
)
lay1_75_5_NJ_ncall200.make_plots()

lay1_75_5_NJ_ncall100 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='all289_NJ_75_5_all13000_ncall100',
    output_dir=output_dir,
    threshold=75, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9
)
lay1_75_5_NJ_ncall100.make_plots()

lay1_75_5_NJ = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='all289_NJ_75_5_all13000',
    output_dir=output_dir,
    threshold=75, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9
)
lay1_75_5_NJ.make_plots()


lay1_75_5_NJ_v2 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='all289_NJ_75_5_all13000_v2',
    output_dir=output_dir,
    threshold=75, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=True
)
lay1_75_5_NJ_v2.make_plots()


lay1_75_10_NJ_v2 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='all289_NJ_75_10_all13000_v2',
    output_dir=output_dir,
    threshold=75, n_trig_thres=10,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=True
)
lay1_75_10_NJ_v2.make_plots()


lay1_75_10_NJ_v2_ncall400 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='all289_NJ_75_10_all13000_v2_ncall400',
    output_dir=output_dir,
    threshold=75, n_trig_thres=10,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=True
)
lay1_75_10_NJ_v2_ncall400.make_plots()



lay1_75_5_NJ_v2_lownoise = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf, layout_name='all289_NJ_75_5_all13000_v2_lownoise',
    output_dir=output_dir,
    threshold=75, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-12, do_swf=True
)
lay1_75_5_NJ_v2_lownoise.make_plots()



idd = np.where(lay1_75_5_NJ_v2.event_res_tab[:, 19]!= -1)[0]
plt.figure(6789)
plt.clf()
plt.scatter(lay1_75_5_NJ_v2.event_res_tab[idd, 16], lay1_75_5_NJ_v2.event_res_tab[idd, 19], c='g', s=3)
plt.scatter(lay1_75_5_NJ_v2.event_res_tab[idd, 16], lay1_75_5_NJ_v2.event_res_tab[idd, 16], c='k', s=2)
plt.scatter(lay1_75_5_NJ_v2.event_res_tab[idd, 16], lay1_75_5_NJ_v2.event_res_tab[idd, 24], c='c', s=2)
plt.scatter(lay1_75_5_NJ_v2.event_res_tab[idd, 16], lay1_75_5_NJ_v2.event_res_tab[idd, 27], c='m', s=2)



### load the xmax proxies if they exist
path_ = '/Users/ab212678/Documents/GRAND/sims/DC2/DC2Training/PWF_SWF_data/'
if os.path.isfile(os.path.join(path_, 'zen_bin_centers.npy')):
    zen_bins_centers = np.load(os.path.join(path_, 'zen_bin_centers.npy'))
    binned_xmax_z = np.load(os.path.join(path_, 'binned_xmax_z_75_4.npy'))
    binned_dmax = np.load(os.path.join(path_, 'binned_dmax_75_4.npy'))

interp_xmax_z = interp.interp1d(zen_bins_centers, binned_xmax_z)
interp_dmax = interp.interp1d(zen_bins_centers, binned_dmax)


idd = np.where(lay1_75_10_NJ_v2.event_res_tab[:, 19]!= -1)[0]
theta_pred = lay1_75_10_NJ_v2.event_res_tab[idd, 10]
phi_pred = lay1_75_10_NJ_v2.event_res_tab[idd, 11]

xmaxz = interp_xmax_z(theta_pred)
dmax = interp_dmax(theta_pred)

xeff = np.cos(phi_pred*D2R) * dmax
yeff = np.sin(phi_pred*D2R) * dmax
zeff = xmaxz


idd = np.where(lay1_75_10_NJ_v2.event_res_tab[:, 19]!= -1)[0]
plt.figure(670)
plt.clf()
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 16], lay1_75_10_NJ_v2.event_res_tab[idd, 19], c='g', s=3)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 16], lay1_75_10_NJ_v2.event_res_tab[idd, 16], c='k', s=2)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 16], lay1_75_10_NJ_v2.event_res_tab[idd, 24], c='c', s=2)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 16], lay1_75_10_NJ_v2.event_res_tab[idd, 27], c='m', s=2)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 16], xeff, c='r', s=2)



idd = np.where(lay1_75_10_NJ_v2.event_res_tab[:, 19]!= -1)[0]
plt.figure(671)
plt.clf()
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 17], lay1_75_10_NJ_v2.event_res_tab[idd, 20], c='g', s=3)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 17], lay1_75_10_NJ_v2.event_res_tab[idd, 17], c='k', s=2)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 17], lay1_75_10_NJ_v2.event_res_tab[idd, 25], c='c', s=2)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 17], lay1_75_10_NJ_v2.event_res_tab[idd, 28], c='m', s=2)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 17], yeff, c='r', s=2)




idd = np.where(lay1_75_10_NJ_v2.event_res_tab[:, 19]!= -1)[0]
plt.figure(672)
plt.clf()
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 18], lay1_75_10_NJ_v2.event_res_tab[idd, 21], c='g', s=3)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 18], lay1_75_10_NJ_v2.event_res_tab[idd, 18], c='k', s=2)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 18], lay1_75_10_NJ_v2.event_res_tab[idd, 26], c='c', s=2)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 18], lay1_75_10_NJ_v2.event_res_tab[idd, 29], c='m', s=2)
plt.scatter(lay1_75_10_NJ_v2.event_res_tab[idd, 18], zeff, c='r', s=2)





idd = np.where(lay1_75_10_NJ_v2_ncall400.event_res_tab[:, 19]!= -1)[0]
plt.figure(680)
plt.clf()
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 16], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 19], c='g', s=3)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 16], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 16], c='k', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 16], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 24], c='c', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 16], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 27], c='m', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 16], xeff, c='r', s=2)



idd = np.where(lay1_75_10_NJ_v2_ncall400.event_res_tab[:, 19]!= -1)[0]
plt.figure(681)
plt.clf()
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 17], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 20], c='g', s=3)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 17], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 17], c='k', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 17], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 25], c='c', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 17], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 28], c='m', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 17], yeff, c='r', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 17], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 17]/2, c='k', s=2)




idd = np.where(lay1_75_10_NJ_v2_ncall400.event_res_tab[:, 19]!= -1)[0]
plt.figure(682)
plt.clf()
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 18], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 21], c='g', s=3)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 18], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 18], c='k', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 18], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 26], c='c', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 18], lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 29], c='m', s=2)
plt.scatter(lay1_75_10_NJ_v2_ncall400.event_res_tab[idd, 18], zeff, c='r', s=2)











idd = np.where(lay1_75_5_NJ_v2_lownoise.event_res_tab[:, 19]!= -1)[0]
plt.figure(6790)
plt.clf()
plt.scatter(lay1_75_5_NJ_v2_lownoise.event_res_tab[idd, 16], lay1_75_5_NJ_v2_lownoise.event_res_tab[idd, 19], c='g', s=3)
plt.scatter(lay1_75_5_NJ_v2_lownoise.event_res_tab[idd, 16], lay1_75_5_NJ_v2_lownoise.event_res_tab[idd, 16], c='k', s=2)
plt.scatter(lay1_75_5_NJ_v2_lownoise.event_res_tab[idd, 16], lay1_75_5_NJ_v2_lownoise.event_res_tab[idd, 24], c='c', s=2)
plt.scatter(lay1_75_5_NJ_v2_lownoise.event_res_tab[idd, 16], lay1_75_5_NJ_v2_lownoise.event_res_tab[idd, 27], c='m', s=2)






plt.scatter(lay1_75_5_NJ_v2.event_res_tab[idd, 16], lay1_75_5_NJ_v2.event_res_tab[idd, 24], c=lay1_75_5_NJ_v2.event_res_tab[idd, 9], vmax = 10)

plt.scatter(lay1_75_5_NJ.event_res_tab[idd, 16], lay1_75_5_NJ.event_res_tab[idd, 19], c='c', s=3)
plt.scatter(lay1_75_5_NJ_ncall200.event_res_tab[idd, 16], lay1_75_5_NJ_ncall200.event_res_tab[idd, 19], c='m', s=3)














plt.figure(100)
plt.clf()
plt.hist(lay1_75_5_NJ.res_phi, bins=50, range=[-3, 3], histtype='step',  label='PWF', density=True)
plt.hist(lay1_75_5_NJ.res_phi_swf, bins=50, range=[-3, 3], histtype='step', label='SWF ncall400', density=True)
plt.hist(lay1_75_5_NJ_v2.res_phi_swf, bins=50, range=[-3, 3], histtype='step', label='SWF  v2 ncall200', density=True)

#plt.hist(lay1_75_5_NJ_ncall200.res_phi_swf, bins=50, range=[-3, 3], histtype='step', label='SWF ncall200')
#plt.hist(lay1_75_5_NJ_ncall100.res_phi_swf, bins =50, range=[-3, 3], alpha=0.4, label='SWF ncall100')
plt.legend()
plt.xlabel('Azimuth residues [deg]')
plt.ylabel('# events')
plt.tight_layout()
plt.savefig('comp_swf_pwf_azimuth.png')



plt.figure(2000)
plt.clf()
plt.hist(lay1_75_5_NJ.res_theta, bins=50, range=[-3, 3], histtype='step',  label='PWF')
plt.hist(lay1_75_5_NJ.res_theta_swf, bins=50, range=[-3, 3], histtype='step', label='SWF ncall400')
#plt.hist(lay1_75_5_NJ_ncall200.res_theta_swf, bins=50, range=[-3, 3], histtype='step', label='SWF ncall200')
#plt.hist(lay1_75_5_NJ_ncall100.res_phi_swf, bins =50, range=[-3, 3], alpha=0.4, label='SWF ncall100')
plt.legend()
plt.xlabel('Zenith residues [deg]')
plt.ylabel('# events')
plt.tight_layout()
plt.savefig('comp_swf_pwf_zenith.png')


plt.figure(3000)
plt.clf()
plt.plot(lay1_75_5_NJ.n_ant, lay1_75_5_NJ_ncall100.res_theta_swf, 'g.', label='ncall=100')
plt.plot(lay1_75_5_NJ.n_ant, lay1_75_5_NJ_ncall200.res_theta_swf, 'b.', label='ncall=200')
plt.plot(lay1_75_5_NJ.n_ant, lay1_75_5_NJ.res_theta_swf, 'r.', label='ncall=400')
plt.plot(lay1_75_5_NJ_v2.n_ant, lay1_75_5_NJ_v2.res_theta_swf, 'c.', label='v2,  ncall=200')

plt.legend()
plt.xlabel('N antennas')
plt.ylabel('zenith residues [deg]')
plt.savefig('res_theta_vs_nants.png')



evt = lay1_75_5_NJ.event_res_tab


is_trigged = np.where(evt[:, 10]> -1)[0]

zen_bins = np.linspace(0, 130, 140)
id_zen = np.digitize(evt[:, 2], zen_bins)


def bin_array(array_x, array_y, array_bins):
    id_bins = np.digitize(array_x, array_bins)
    n_bins = len(array_bins) - 1

    mean_values = np.zeros(n_bins)
    std_values = np.zeros(n_bins)
    n_values = np.zeros(n_bins)

    for i in range(0, n_bins):
        idd = np.where(id_bins == i+1)[0]
        if len(idd) > 0:
            mean_values[i] = array_y[idd].mean()
            std_values[i] = array_y[idd].std()

            n_values[i] = len(array_y[idd])

    return mean_values, std_values, n_values



zen_bins_centers = (zen_bins[1:]+zen_bins[:-1])/2


plt.figure(1)
plt.clf()
plt.scatter(evt[:, 2], evt[:, 3], c=evt[:, 18])
#plt.scatter(evt[is_trigged, 2], evt[is_trigged, 3], c=evt[is_trigged, 18])

plt.xlabel('zenith [deg]')
plt.ylabel('azimuth [deg]')
plt.colorbar(label='xmax_z [m]')
plt.title('xmax altitude')
plt.tight_layout()
plt.savefig('scatter_xmax_z.png')


res18 = bin_array(evt[:, 2], evt[:, 18], zen_bins)
res18_trigged = bin_array(evt[is_trigged, 2], evt[is_trigged, 18], zen_bins)
plt.figure(2)
plt.clf()
plt.plot(evt[:, 2], evt[:, 18], 'k.')
plt.plot(evt[is_trigged, 2], evt[is_trigged, 18], 'r.')

plt.plot(zen_bins_centers, res18[0], 'b-')
plt.plot(zen_bins_centers, res18_trigged[0], 'g-')
plt.title('xmax altitude vs zenith')
plt.xlabel('zenith [deg]')
plt.ylabel('xmax altitude [m]')
plt.tight_layout()
plt.savefig('xmax_z_vs_zenith.png')



d2d_max = np.sqrt((evt[:, 17] - 0*evt[:, 5])**2 + (evt[:, 16] - 0*evt[:, 4])**2)
plt.figure(3)
plt.clf()
plt.scatter(evt[:, 2], evt[:, 3], c=d2d_max)
plt.title('2d distance from sc to xmax')
plt.xlabel('zenith [deg]')
plt.ylabel('azimuth [deg]')
plt.colorbar(label='d2d_max [m]')

d2d_max = np.sqrt((evt[:, 17] - 0*evt[:, 5])**2 + (evt[:, 16] - 0*evt[:, 4])**2)
res_dmax = bin_array(evt[:, 2], d2d_max, zen_bins)
res_dmax_trigged = bin_array(evt[is_trigged, 2], d2d_max[is_trigged], zen_bins)


plt.figure(4)
plt.clf()
plt.plot(evt[:, 2], d2d_max, 'k.')
plt.plot(evt[is_trigged, 2], d2d_max[is_trigged], 'r.')

plt.plot(zen_bins_centers, res_dmax[0], 'b-')
plt.plot(zen_bins_centers, res_dmax_trigged[0], 'g-')
plt.xlim(25, 92)
plt.title('2d distance from sc to xmax')
plt.xlabel('zenith [deg]')
plt.ylabel('d2d_max [m]')
plt.tight_layout()
plt.savefig('dmax_vs_zenith.png')





    #lay1_75_4_NJ.make_plots()


    # lay_gp13_75_4_NJ = ldu.Layout_dc2(
    #     du_pos_all, du_names_all, data_dir_zhaires_nj, layout_name='GP13_NJ_75_4',
    #     output_dir=output_dir,
    #     du_names=gp13_id, threshold=75, n_trig_thres=4
    #     )
    # #lay_gp13_75_4_NJ.make_plots()


    # lay_1000_75_4_NJ = ldu.Layout_dc2(
    #     du_pos_all,
    #     du_names_all, data_dir_zhaires_nj,
    #     output_dir=output_dir,
    #     layout_name='coarse1000_NJ_75_4', du_names=gp1000_names, threshold=75, n_trig_thres=4
    # )
    # #lay_1000_75_4_NJ.make_plots()




    # lay_infill2 = ldu.Layout_dc2(
    #     du_pos_all, du_names_all,
    #     data_dir_zhaires_nj, du_names=ids_infill, layout_name='infill2_NJ_75_4',
    #     threshold=75, n_trig_thres=4,
    #     output_dir=output_dir,
    #     do_noise_timing=True,
    #     sigma_timing=5e-9
    # )
    # #lay_infill2.make_plots()




    # binning1 = Binning()


    # lay_gaa_75_3_Cc=ldu.Layout_dc2(
    #     du_pos_all_gaa, du_names_all_gaa,
    #     data_gaa, layout_name='gaa_NJ_75_3_CC',
    #     output_dir=output_dir,
    #     threshold=75, n_trig_thres=3,
    #     do_noise_timing=True,
    #     sigma_timing=5e-9
    # )
    # #lay_gaa_75_3_CC.make_plots()


    # lay_gaa_75_3_nCc=ldu.Layout_dc2(
    #     du_pos_all_gaa, du_names_all_gaa,
    #     data_gaa, layout_name='gaa_NJ_75_3_nCC',
    #     threshold=75, n_trig_thres=3,
    #     do_noise_timing=True,
    #     sigma_timing=5e-9
    # )
    # #lay_gaa_75_3_nCC.make_plots()



