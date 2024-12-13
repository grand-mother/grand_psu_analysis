

import numpy as np
import glob
import matplotlib.pyplot as plt

# better plots
from matplotlib import rc

import os
from grid_shape_lib.modules import masks as masks
from grand_psu_lib.modules import layout_dc2 as ldu


# rc('font', **{'family':'serif','serif':['Palatino']})
# rc('text', usetex = True)
rc('font', size = 16.0)

D2R = np.pi / 180
R2D = 180 / np.pi


sigma_adc = 4
sigma_ef = 8




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

data_pwf_swf_nonoise = '/Users/ab212678/Documents/GRAND/sims/DC2/DC2Training/files_DC2_ZHAireS-NJ_nonoise_adc/'
data_pwf_swf_withnoise = '/Users/ab212678/Documents/GRAND/sims/DC2/DC2Training/files_DC2_ZHAireS_noise_adc/'


data_dir_zhaires_nj = '/Users/ab212678/Documents/GRAND/sims/DC2/DC2Training/PWFdata_3june24/Zhaires_NJ'


output_dir = '/Users/ab212678/Documents/GRAND/Codes/grand_psu_analysis/study_pwf_swf_dec2024/'



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








# baseline 
l1_100_5_nonoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nonoise, layout_name='l1_100_5_nonoise_ef_pwf',
    output_dir=output_dir,
    threshold=100, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l1_100_5_nonoise_ef_pwf.make_plots()


# impact of noise and threshold
l1_100_5_withnoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_100_5_withnoise_ef_pwf',
    output_dir=output_dir,
    threshold=100, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l1_100_5_withnoise_ef_pwf.make_plots()



#vary the threshold
l1_50_5_withnoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_50_5_withnoise_ef_pwf',
    output_dir=output_dir,
    threshold=50, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l1_50_5_withnoise_ef_pwf.make_plots()


l1_25_5_withnoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_25_5_withnoise_ef_pwf',
    output_dir=output_dir,
    threshold=25, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l1_25_5_withnoise_ef_pwf.make_plots()


# plt.figure()
# plt.hist(l1_100_5_nonoise_ef_pwf.res_phi, bins=50, histtype='step', range=[-2, 2], density=True)
# plt.hist(l1_100_5_withnoise_ef_pwf.res_phi, bins=50, histtype='step', range=[-2, 2], density=True)
# plt.hist(l1_75_5_withnoise_ef_pwf.res_phi, bins=50, histtype='step', range=[-2, 2], density=True)
# plt.hist(l1_50_5_withnoise_ef_pwf.res_phi, bins=50, histtype='step', range=[-2, 2], density=True)
# plt.hist(l1_25_5_withnoise_ef_pwf.res_phi, bins=50, histtype='step', range=[-2, 2], density=True)


# meme chose avec les adc
# baseline 
l1_100_5_nonoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nonoise, layout_name='l1_100_5_nonoise_tadc_pwf',
    output_dir=output_dir,
    threshold=100, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l1_100_5_nonoise_tadc_pwf.make_plots()


# impact of noise and threshold
l1_100_5_withnoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_100_5_withnoise_tadc_pwf',
    output_dir=output_dir,
    threshold=100, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l1_100_5_withnoise_tadc_pwf.make_plots()

l1_25_5_withnoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_25_5_withnoise_tadc_pwf',
    output_dir=output_dir,
    threshold=25, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l1_25_5_withnoise_tadc_pwf.make_plots()


l1_25_5_nonoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_25_5_nonoise_tadc_pwf',
    output_dir=output_dir,
    threshold=25, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l1_25_5_nonoise_tadc_pwf.make_plots()




l1_22_5_withnoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_22_5_withnoise_tadc_pwf',
    output_dir=output_dir,
    threshold=22, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l1_22_5_withnoise_tadc_pwf.make_plots()



l1_20_5_withnoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_20_5_withnoise_tadc_pwf',
    output_dir=output_dir,
    threshold=20, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l1_20_5_withnoise_tadc_pwf.make_plots()



plt.figure(2)
plt.clf()
plt.hist(l1_100_5_nonoise_tadc_pwf.res_phi, bins=100, histtype='step', range=[-2, 2], density=True)
plt.hist(l1_100_5_withnoise_tadc_pwf.res_phi, bins=100, histtype='step', range=[-2, 2], density=True)
plt.hist(l1_25_5_withnoise_tadc_pwf.res_phi, bins=100, histtype='step', range=[-2, 2], density=True)
plt.hist(l1_25_5_nonoise_tadc_pwf.res_phi, bins=100, histtype='step', range=[-2, 2], density=True)
plt.hist(l1_22_5_withnoise_tadc_pwf.res_phi, bins=100, histtype='step', range=[-2, 2], density=True)
plt.hist(l1_20_5_withnoise_tadc_pwf.res_phi, bins=100, histtype='step', range=[-2, 2], density=True)


plt.figure(3)
plt.clf()
plt.hist(l1_100_5_nonoise_tadc_pwf.res_theta, bins=100, histtype='step', range=[-2, 2], density=True)
plt.hist(l1_100_5_withnoise_tadc_pwf.res_theta, bins=100, histtype='step', range=[-2, 2], density=True)
plt.hist(l1_25_5_withnoise_tadc_pwf.res_theta, bins=100, histtype='step', range=[-2, 2], density=True)
plt.hist(l1_22_5_withnoise_tadc_pwf.res_theta, bins=100, histtype='step', range=[-2, 2], density=True)
plt.hist(l1_20_5_withnoise_tadc_pwf.res_theta, bins=100, histtype='step', range=[-2, 2], density=True)




### run infill500 l2


# baseline 
l2_100_5_nonoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nonoise, du_names=ids_infill500, layout_name='l2_100_5_nonoise_ef_pwf',
    output_dir=output_dir,
    threshold=100, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l2_100_5_nonoise_ef_pwf.make_plots()


# impact of noise and threshold
l2_100_5_withnoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, du_names=ids_infill500, layout_name='l2_100_5_withnoise_ef_pwf',
    output_dir=output_dir,
    threshold=100, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l2_100_5_withnoise_ef_pwf.make_plots()


l2_50_5_withnoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, du_names=ids_infill500, layout_name='l2_50_5_withnoise_ef_pwf',
    output_dir=output_dir,
    threshold=50, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l2_50_5_withnoise_ef_pwf.make_plots()

l2_25_5_withnoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, du_names=ids_infill500, layout_name='l2_25_5_withnoise_ef_pwf',
    output_dir=output_dir,
    threshold=25, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l2_25_5_withnoise_ef_pwf.make_plots()



# meme chose avec les adc
# baseline 
l2_100_5_nonoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nonoise, du_names=ids_infill500, layout_name='l2_100_5_nonoise_tadc_pwf',
    output_dir=output_dir,
    threshold=100, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l2_100_5_nonoise_tadc_pwf.make_plots()


# impact of noise and threshold
l2_100_5_withnoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, du_names=ids_infill500, layout_name='l2_100_5_withnoise_tadc_pwf',
    output_dir=output_dir,
    threshold=100, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l2_100_5_withnoise_tadc_pwf.make_plots()


l2_25_5_withnoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, du_names=ids_infill500, layout_name='l2_25_5_withnoise_tadc_pwf',
    output_dir=output_dir,
    threshold=25, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l2_25_5_withnoise_tadc_pwf.make_plots()


# l2_25_5_withnoise_tadc_swf = ldu.Layout_dc2(
#     du_pos_all, du_names_all,
#     data_pwf_swf_withnoise, du_names=ids_infill500, layout_name='l2_25_5_withnoise_tadc_swf',
#     output_dir=output_dir,
#     threshold=25, n_trig_thres=5,
#     do_noise_timing=False,
#     sigma_timing=5e-9, do_swf=True, qty_to_use='tadc', ncall=4000
# )
# l2_25_5_withnoise_tadc_swf.make_plots()

l1_26_5_withnoise_tadc_swf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_26_5_withnoise_tadc_swf',
    output_dir=output_dir,
    threshold=26, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=True, qty_to_use='tadc', ncall=4000
)
l1_26_5_withnoise_tadc_swf.make_plots()


l1_26_5_withnoise_tadc_swf_ncall200 = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_26_5_withnoise_tadc_swf_ncall200',
    output_dir=output_dir,
    threshold=26, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=True, qty_to_use='tadc', ncall=200
)
l1_26_5_withnoise_tadc_swf_ncall200.make_plots()

l1_25_5_withnoise_tadc_swf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_25_5_withnoise_tadc_swf',
    output_dir=output_dir,
    threshold=25, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=True, qty_to_use='tadc', ncall=4000
)
l1_25_5_withnoise_tadc_swf.make_plots()


l1_50_5_withnoise_tadc_swf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_50_5_withnoise_tadc_swf',
    output_dir=output_dir,
    threshold=50, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=True, qty_to_use='tadc', ncall=4000
)
l1_50_5_withnoise_tadc_swf.make_plots()


k_tab = ldu.swf.utils.thetaphi_to_k(la.theta_gt*D2R, la.phi_gt*D2R)


### make a plot about swf fit
# event_res_tab[k, 40] = swf_fit_v4[0]
# event_res_tab[k, 41] = swf_fit_v4[1]
# event_res_tab[k, 42] = swf_fit_v4[2]

# event_res_tab[k, 30] = xeff
# event_res_tab[k, 31] = yeff
# event_res_tab[k, 32] = zeff

# xmax_x = event_res_tab[k, 16]
# xmax_y = event_res_tab[k, 17]
# xmax_z = event_res_tab[k, 18]



l2_22_5_withnoise_tadc_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, du_names=ids_infill500, layout_name='l2_22_5_withnoise_tadc_pwf',
    output_dir=output_dir,
    threshold=22, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc', ncall=2000
)
l2_22_5_withnoise_tadc_pwf.make_plots()


l1_110_10_withnoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, layout_name='l1_110_10_withnoise_ef_pwf',
    output_dir=output_dir,
    threshold=110, n_trig_thres=10,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l1_110_10_withnoise_ef_pwf.make_plots()

l2_110_10_withnoise_ef_pwf = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_withnoise, du_names=ids_infill500, layout_name='l2_110_10_withnoise_ef_pwf',
    output_dir=output_dir,
    threshold=110, n_trig_thres=10,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef', ncall=2000
)
l2_110_10_withnoise_ef_pwf.make_plots()






# plt.figure(4)
# plt.clf()
# plt.hist(l2_100_5_nonoise_ef_pwf.res_phi, bins=30, histtype='step', range=[-2, 2], density=True)
# plt.hist(l2_100_5_withnoise_ef_pwf.res_phi, bins=30, histtype='step', range=[-2, 2], density=True)
# plt.hist(l2_100_5_nonoise_tadc_pwf.res_phi, bins=30, histtype='step', range=[-2, 2], density=True)
# plt.hist(l2_100_5_withnoise_tadc_pwf.res_phi, bins=30, histtype='step', range=[-2, 2], density=True)





# plt.figure(5)
# plt.clf()
# plt.hist(l2_100_5_nonoise_ef_pwf.res_theta, bins=30, histtype='step', range=[-2, 2], density=True)
# plt.hist(l2_100_5_withnoise_ef_pwf.res_theta, bins=30, histtype='step', range=[-2, 2], density=True)
# plt.hist(l2_100_5_nonoise_tadc_pwf.res_theta, bins=30, histtype='step', range=[-2, 2], density=True)
# plt.hist(l2_100_5_withnoise_tadc_pwf.res_theta, bins=30, histtype='step', range=[-2, 2], density=True)


## histograms of the ang errors on ef with different threshold to knonw the cut 

plt.figure(3465)
plt.clf()
plt.hist(l1_100_5_withnoise_ef_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='100 muV/M, {}events'.format(l1_100_5_withnoise_ef_pwf.res_phi.shape[0]))
plt.hist(l1_50_5_withnoise_ef_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='50 muV/M, {}events'.format(l1_50_5_withnoise_ef_pwf.res_phi.shape[0]))
plt.hist(l1_25_5_withnoise_ef_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='25 muV/M, {}events'.format(l1_25_5_withnoise_ef_pwf.res_phi.shape[0]))
plt.legend()
plt.title('full layout EF')
plt.xlabel('Angular error [deg]')
plt.ylabel('Density')

plt.figure(3466)
plt.clf()
plt.hist(l1_100_5_withnoise_tadc_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='100 adc, {}events'.format(l1_100_5_withnoise_tadc_pwf.res_phi.shape[0]))
plt.hist(l1_25_5_withnoise_tadc_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='25 adc, {}events'.format(l1_25_5_withnoise_tadc_pwf.res_phi.shape[0]))
plt.hist(l1_22_5_withnoise_tadc_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='22 adc, {}events'.format(l1_22_5_withnoise_tadc_pwf.res_phi.shape[0]))
plt.hist(l1_20_5_withnoise_tadc_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='20 adc, {}events'.format(l1_20_5_withnoise_tadc_pwf.res_phi.shape[0]))
plt.legend()
plt.title('full layout ADC')
plt.xlabel('Angular error [deg]')
plt.ylabel('Density')

### Work with l2 layout


plt.figure(3467)
plt.clf()
plt.hist(l2_100_5_withnoise_ef_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='100 muV/M, {}events'.format(l2_100_5_withnoise_ef_pwf.res_phi.shape[0]))
plt.hist(l2_50_5_withnoise_ef_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='50 muV/M, {}events'.format(l2_50_5_withnoise_ef_pwf.res_phi.shape[0]))
#plt.hist(l2_25_5_withnoise_ef_pwf.ang_err*R2D, bins=50, range=[0, 1], histtype='step', lw=2, density=True, label='25 muV/M, {}events'.format(l2_25_5_withnoise_ef_pwf.res_phi.shape[0]))
plt.legend()
plt.title('500m infill layout EF')
plt.xlabel('Angular error [deg]')
plt.ylabel('Density')

plt.figure(3468)
plt.clf()
plt.hist(l2_100_5_withnoise_tadc_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='100 adc, {}events'.format(l2_100_5_withnoise_tadc_pwf.res_phi.shape[0]))
plt.hist(l2_25_5_withnoise_tadc_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='25 adc, {}events'.format(l2_25_5_withnoise_tadc_pwf.res_phi.shape[0]))
plt.hist(l2_22_5_withnoise_tadc_pwf.ang_err*R2D, bins=50, range=[0, 2], histtype='step', lw=2, density=True, label='22 adc, {}events'.format(l2_22_5_withnoise_tadc_pwf.res_phi.shape[0]))
plt.legend()
plt.title('500m infill layout ADC')
plt.xlabel('Angular error [deg]')
plt.ylabel('Density')
























# lay_infill500_75_5_nonoise_ef = ldu.Layout_dc2(
#     du_pos_all, du_names_all,
#     data_pwf_swf_nonoise, du_names=ids_infill500, layout_name='infill500_nonoise_75_5_ef',
#     threshold=75, n_trig_thres=5,
#     output_dir=output_dir,
#     do_noise_timing=True,
#     sigma_timing=5e-9, do_swf=True, qty_to_use='ef', ncall=2000
# )
# lay_infill500_75_5_nonoise_ef.make_plots()




# def get_sublayout_names(sublayout_pos, main_lay_pos, main_lay_names):
#     ids = []
#     for pos in sublayout_pos:
#         ids.append(int(masks.get_closest_antenna(pos[0], pos[1], main_lay_pos[:, 0:2].T )))

#     return list(np.int32(main_lay_names[ids]))







# lay1_6_5_nonoise_tadc = ldu.Layout_dc2(
#     du_pos_all, du_names_all,
#     data_pwf_swf_nonoise, layout_name='all289_nonoise_6_5_tadc',
#     output_dir=output_dir,
#     threshold=6, n_trig_thres=5,
#     do_noise_timing=True,
#     sigma_timing=5e-9, do_swf=True, qty_to_use='tadc', ncall=2000
# )
# lay1_6_5_nonoise_tadc.make_plots()


# lay1_6_5_withnoise_tadc = ldu.Layout_dc2(
#     du_pos_all, du_names_all,
#     data_pwf_swf_withnoise, layout_name='all289_withnoise_6_5_tadc',
#     output_dir=output_dir,
#     threshold=6, n_trig_thres=5,
#     do_noise_timing=False,
#     sigma_timing=5e-9, do_swf=True, qty_to_use='tadc', ncall=2000
# )
# lay1_6_5_withnoise_tadc.make_plots()



# lay = lay1_75_5_nonoise_ef
# idg = np.where(lay.event_res_tab[:, 10]> -1)[0]


# plt.figure(1)
# plt.clf()
# plt.scatter(lay.event_res_tab[idg, 18], lay.event_res_tab[idg, 42]- lay.event_res_tab[idg, 18], s=5, c=lay.event_res_tab[idg, 9], cmap='Paired' )
# plt.colorbar(label='# antennas')



# plt.figure(2)
# plt.clf()
# plt.hist(lay.event_res_tab[idg, 16]-lay.event_res_tab[idg, 40], bins=100, histtype='step' )
# plt.hist(lay.event_res_tab[idg, 17]-lay.event_res_tab[idg, 41], bins=100, histtype='step' )
# plt.figure(3)
# plt.clf()
# plt.hist(lay.event_res_tab[idg, 18]-lay.event_res_tab[idg, 42], bins=100, histtype='step' )




# plt.figure(56)
# plt.clf()
# #plt.hist(lay1_25_5_nj_ef.res_phi, bins=30, range=[-3, 3], alpha=0.5, label='25muV/m')
# #plt.hist(lay1_30_5_nj_ef.res_phi, bins=30, range=[-3, 3], alpha=0.5, label='30muV/m')
# #plt.hist(lay1_35_5_nj_ef.res_phi, bins=30, range=[-3, 3], alpha=0.5, label='35muV/m')
# plt.hist(lay1_40_5_nj_ef.res_phi, bins=30, range=[-3, 3], alpha=0.5, label='40muV/m', density=True)
# plt.hist(lay1_45_5_nj_ef.res_phi, bins=30, range=[-3, 3], alpha=0.5, label='45muV/m', density=True)
# plt.hist(lay1_50_5_nj_ef.res_phi, bins=30, range=[-3, 3], alpha=0.5, label='50muV/m', density=True)
# plt.hist(lay1_75_5_nj_ef.res_phi, bins=30, range=[-3, 3], alpha=0.5, label='75muV/m', density=True)
# plt.legend()


# plt.figure(57)
# plt.clf()
# #plt.hist(lay1_25_5_nj_ef.res_theta, bins=30, range=[-3, 3], alpha=0.5, label='25muV/m')
# #plt.hist(lay1_30_5_nj_ef.res_theta, bins=30, range=[-3, 3], alpha=0.5, label='30muV/m')
# #plt.hist(lay1_35_5_nj_ef.res_theta, bins=30, range=[-3, 3], alpha=0.5, label='35muV/m')
# plt.hist(lay1_40_5_nj_ef.res_theta, bins=30, range=[-3, 3], alpha=0.5, label='40muV/m', density=True)
# plt.hist(lay1_45_5_nj_ef.res_theta, bins=30, range=[-3, 3], alpha=0.5, label='45muV/m', density=True)

# plt.hist(lay1_50_5_nj_ef.res_theta, bins=30, range=[-3, 3], alpha=0.5, label='50muV/m', density=True)
# plt.hist(lay1_75_5_nj_ef.res_theta, bins=30, range=[-3, 3], alpha=0.5, label='75muV/m', density=True)
# plt.legend()
