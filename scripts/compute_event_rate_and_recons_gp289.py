
import numpy as np
import glob
import matplotlib.pyplot as plt

# better plots
from matplotlib import rc

import os
from grid_shape_lib.modules import masks as masks
from grand_psu_lib.modules import layout_dc2 as ldu


### compute the event for the real gp300/gp80 tentative layouts using the DC2-like sims made by matias on the the real position of the gp300. i.e. not the DC2 layout.

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





def load_all_antennas_GP299():
    # this event contains all the 299 antennas
    ev_id = 21
    res_file = os.path.join(data_dir_GP299, 'data_files/{}.npy'.format(ev_id))

    arr = np.load(res_file)
    du_names = arr[:, 0]
    du_pos = arr[:, 1:4]

    return du_pos, du_names



data_dir_GP299 = '/Users/ab212678/Documents/GRAND/sims/sim_gp289_d2like/sim_dc2_299_extraction/'

output_dir = './sim_dc2_299_eventrates'

du_pos_all_gp299, du_names_all_gp299 = load_all_antennas_GP299()

### reload the .txt meters and redo the plots to make sure
gp300_xy = np.loadtxt('./gp80_positions/gp300_official_position_meters.txt', skiprows=1)
gp80_coarse = np.loadtxt('gp80_positions/gp80_coarse_position_meters.txt', skiprows=1)
gp80_infill = np.loadtxt('gp80_positions/gp80_infill_position_meters.txt', skiprows=1)
gp80_ellip = np.loadtxt('gp80_positions/gp80_elliptical_position_meters.txt', skiprows=1)
gp80_hybrid = np.loadtxt('gp80_positions/gp80_hybrid_position_meters.txt', skiprows=1)

#gp80_ellip3 = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 78, 81, 82, 85, 88, 91, 93, 102, 105, 116, 119]
gp_80_ellip3 = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 39, 40, 41, 42, 43, 48, 49, 50, 51, 52, 53, 54, 55, 75, 77, 78, 79, 80, 81, 83, 85, 89, 91, 92, 93, 95, 97, 103, 105, 107, 109, 111, 113, 129, 131, 137, 139, 141, 143]) -1

gp_80_ellip3 = np.array([84, 50, 38, 49, 83, 90, 54, 34, 26, 25, 33, 53, 89, 104, 42, 22, 18, 12, 17, 21, 41, 103, 118, 80, 46, 30, 10, 4, 9, 29, 45, 79, 117, 92, 36, 14, 6, 0, 5, 13, 35, 91, 115, 77, 43, 27, 7, 1, 2, 8, 28, 44, 78, 116, 101, 39, 19, 15, 11, 16, 20, 40, 102, 87, 51, 31, 23, 24, 32, 52, 88, 81, 47, 37, 48, 82])



plt.figure(35, figsize=(20, 10))
plt.clf()
plt.plot(-gp300_xy[:, 1], gp300_xy[:, 0], 'co', ms=7, alpha=0.4, label='GP300')
plt.plot(-gp80_coarse[:, 1], gp80_coarse[:, 0], 'k*', ms=10, alpha=0.4, label='GP80 coarse')
plt.plot(-gp80_ellip[:, 1], gp80_ellip[:, 0], 'bo', ms=12, fillstyle='none', label='GP80 elliptic')
plt.plot(-gp80_hybrid[:, 1], gp80_hybrid[:, 0], 'orange', ls = 'None', marker='.', ms=11, label='GP80 hybrid')
plt.plot(-gp80_infill[:, 1], gp80_infill[:, 0], 'rh', ms=2, label='GP80 infill')

for i in range(299):
    plt.text(-gp300_xy[i, 1], gp300_xy[i, 0], '{}'.format(int(i)))
plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.title('GP300 and GP80 candidates positions ')
plt.axis('equal')
plt.legend()
plt.tight_layout()




def get_sublayout_names(sublayout_pos, main_lay_pos, main_lay_names):
    ids = []
    for pos in sublayout_pos:
        ids.append(int(masks.get_closest_antenna(pos[0], pos[1], main_lay_pos[:, 0:2].T )))

    return list(np.int32(main_lay_names[ids]))


names_ellip = get_sublayout_names(gp80_ellip, du_pos_all_gp299, du_names_all_gp299)
names_coarse = get_sublayout_names(gp80_coarse, du_pos_all_gp299, du_names_all_gp299)
names_hybrid = get_sublayout_names(gp80_hybrid, du_pos_all_gp299, du_names_all_gp299)
names_infill = get_sublayout_names(gp80_infill, du_pos_all_gp299, du_names_all_gp299)

names_ellip4 = gp_80_ellip3



lay_gp300_75_5 = ldu.Layout_dc2(
        du_pos_all_gp299, du_names_all_gp299,
        data_dir_GP299, layout_name='gp300_75_5',
        output_dir=output_dir,
        du_names=list(np.int32(du_names_all_gp299)),
        threshold=75, n_trig_thres=5,
        do_noise_timing=True,
        sigma_timing=5e-9,
        is_coreas=False
)
lay_gp300_75_5.make_plots()




lay_gp80hexa_75_5 = ldu.Layout_dc2(
        du_pos_all_gp299, du_names_all_gp299,
        data_dir_GP299, layout_name='gp80hexa_75_5',
        output_dir=output_dir,
        du_names=names_hybrid,
        threshold=75, n_trig_thres=5,
        do_noise_timing=True,
        sigma_timing=5e-9,
        is_coreas=False
)
lay_gp80hexa_75_5.make_plots()



lay_gp80coarse_75_5 = ldu.Layout_dc2(
        du_pos_all_gp299, du_names_all_gp299,
        data_dir_GP299, layout_name='gp80coarse_75_5',
        output_dir=output_dir,
        du_names=names_coarse,
        threshold=75, n_trig_thres=5,
        do_noise_timing=True,
        sigma_timing=5e-9,
        is_coreas=False
)
lay_gp80coarse_75_5.make_plots()


lay_gp80ellip_75_5 = ldu.Layout_dc2(
        du_pos_all_gp299, du_names_all_gp299,
        data_dir_GP299, layout_name='gp80ellip_75_5',
        output_dir=output_dir,
        du_names=names_ellip,
        threshold=75, n_trig_thres=5,
        do_noise_timing=True,
        sigma_timing=5e-9,
        is_coreas=False
)
lay_gp80ellip_75_5.make_plots()


lay_gp80ellip4_75_5 = ldu.Layout_dc2(
        du_pos_all_gp299, du_names_all_gp299,
        data_dir_GP299, layout_name='gp80ellip4_75_5',
        output_dir=output_dir,
        du_names=list(names_ellip4),
        threshold=75, n_trig_thres=5,
        do_noise_timing=True,
        sigma_timing=5e-9,
        is_coreas=False
)
lay_gp80ellip4_75_5.make_plots()




lay_gp80infill_75_5 = ldu.Layout_dc2(
        du_pos_all_gp299, du_names_all_gp299,
        data_dir_GP299, layout_name='gp80infill_75_5',
        output_dir=output_dir,
        du_names=names_infill,
        threshold=75, n_trig_thres=5,
        do_noise_timing=True,
        sigma_timing=5e-9,
        is_coreas=False
)
lay_gp80infill_75_5.make_plots()





binning1 = Binning(n_bin_energy=10, n_bin_zen=8, energy_min=16.5, zen_min=50, zen_max=90)

#compute_effective_area(self, binning, cst_A_Sim=None, varing_area=False, core_contained=False):

lay_gp80hexa_75_5.compute_effective_area(binning1, varing_area=True)
lay_gp80hexa_75_5.compute_event_rate(binning1)
#lay_gp80hexa_75_5.plot_binned_quantities(binning1)

lay_gp80ellip_75_5.compute_effective_area(binning1, varing_area=True)
#lay_gp80ellip_75_5.plot_binned_quantities(binning1)
lay_gp80ellip_75_5.compute_event_rate(binning1)


lay_gp80infill_75_5.compute_effective_area(binning1, varing_area=True)
lay_gp80infill_75_5.compute_event_rate(binning1)
#lay_gp80infill_75_5.plot_binned_quantities(binning1)

lay_gp80coarse_75_5.compute_effective_area(binning1, varing_area=True)
#lay_gp80coarse_75_5.plot_binned_quantities(binning1)
lay_gp80coarse_75_5.compute_event_rate(binning1)



lay_gp80ellip4_75_5.compute_effective_area(binning1, varing_area=True)
#lay_gp80ellip4_75_5.plot_binned_quantities(binning1)
lay_gp80ellip4_75_5.compute_event_rate(binning1)





lay_gp300_75_5.compute_effective_area(binning1, varing_area=True)
#lay_gp300_75_5.plot_binned_quantities(binning1)
lay_gp300_75_5.compute_event_rate(binning1)


lays = [
    lay_gp80hexa_75_5, lay_gp80ellip_75_5, lay_gp80ellip4_75_5
]
for lay in lays:
    print('Total number of event per day for {} = {}\n'.format(lay.layout_name, lay.rate_per_day_per_m2.sum()))





plt.figure(figsize=(10, 8))
#plt.stairs(lay_gp80coarse_75_5.rate_per_day_per_m2.sum(axis=0), binning1.energy_bin_edges, color='C3', ls='--', lw=2, label='lay_gp80coarse_75_5')
#plt.stairs(lay_gp80infill_75_5.rate_per_day_per_m2.sum(axis=0), binning1.energy_bin_edges, color='C4', ls='--', lw=2, label='lay_gp80infill_75_5')
plt.stairs(lay_gp80ellip_75_5.rate_per_day_per_m2.sum(axis=0), binning1.energy_bin_edges, color='C1', ls='--', lw=2, label='lay_gp80ellip_75_5')
plt.stairs(lay_gp80hexa_75_5.rate_per_day_per_m2.sum(axis=0), binning1.energy_bin_edges, color='C2', ls='--', lw=2, label='lay_gp80hexa_75_5')
plt.stairs(lay_gp80ellip4_75_5.rate_per_day_per_m2.sum(axis=0), binning1.energy_bin_edges, color='C5', ls=':', lw=2, label='lay_gp80ellip4_75_5')

#plt.stairs(lay_gp300_75_5.rate_per_day_per_m2.sum(axis=0), binning1.energy_bin_edges, color='C5', ls='--', lw=2, label='lay_gp300_75_5')
plt.legend()
plt.yscale('log')
plt.ylabel('#Event per day')
plt.xlabel('log10 Energy/eV')

plt.savefig('eventrate_vs_energy.png')

plt.figure(figsize=(10, 8))

#plt.stairs(lay_gp80coarse_75_5.rate_per_day_per_m2.sum(axis=1), binning1.zen_bin_edges, color='C3', ls='--', lw=2, label='lay_gp80coarse_75_5')
#plt.stairs(lay_gp80infill_75_5.rate_per_day_per_m2.sum(axis=1), binning1.zen_bin_edges, color='C4', ls='--', lw=2, label='lay_gp80infill_75_5')
#plt.stairs(lay_gp300_75_5.rate_per_day_per_m2.sum(axis=1), binning1.zen_bin_edges, color='C5', ls='--', lw=2, label='lay_gp300_75_5')
plt.stairs(lay_gp80ellip_75_5.rate_per_day_per_m2.sum(axis=1), binning1.zen_bin_edges, color='C1', ls='--', lw=2, label='lay_gp80ellip_75_5')
plt.stairs(lay_gp80hexa_75_5.rate_per_day_per_m2.sum(axis=1), binning1.zen_bin_edges, color='C2', ls='--', lw=2, label='lay_gp80hexa_75_5')
plt.stairs(lay_gp80ellip4_75_5.rate_per_day_per_m2.sum(axis=1), binning1.zen_bin_edges, color='C5', ls=':', lw=2, label='lay_gp80ellip4_75_5')


plt.legend()
plt.yscale('log')
plt.ylabel('#Event per day')
plt.xlabel('Zenith [deg]')

plt.savefig('eventrate_vs_zenith.png')



plt.figure()
plt.plot(lay_gp80hexa_75_5.phi_gt, lay_gp80hexa_75_5.res_phi, 'k.')
plt.plot(lay_gp80ellip_75_5.phi_gt, lay_gp80ellip_75_5.res_phi, 'r.')

plt.figure()
plt.plot(lay_gp80hexa_75_5.theta_gt, lay_gp80hexa_75_5.res_phi, 'k.')
plt.plot(lay_gp80ellip_75_5.theta_gt, lay_gp80ellip_75_5.res_phi, 'r.')


plt.figure()
plt.plot(lay_gp80hexa_75_5.phi_gt, lay_gp80hexa_75_5.res_theta, 'k.')
plt.plot(lay_gp80ellip_75_5.phi_gt, lay_gp80ellip_75_5.res_theta, 'r.')
plt.plot(lay_gp80ellip4_75_5.phi_gt, lay_gp80ellip4_75_5.res_theta, 'c.')

plt.figure()
plt.plot(lay_gp80hexa_75_5.theta_gt, lay_gp80hexa_75_5.res_theta, 'k.')
plt.plot(lay_gp80ellip_75_5.theta_gt, lay_gp80ellip_75_5.res_theta, 'r.')
plt.plot(lay_gp80ellip4_75_5.theta_gt, lay_gp80ellip4_75_5.res_theta, 'c.')




plt.figure()
plt.hist(lay_gp80hexa_75_5.res_phi[lay_gp80hexa_75_5.theta_gt>80], density=True, histtype='step', bins=15, range=[-1, 1], label='hexa')
plt.hist(lay_gp80ellip_75_5.res_phi[lay_gp80ellip_75_5.theta_gt>80], density=True, histtype='step', bins=15, range=[-1, 1], label='ellip')
plt.hist(lay_gp80ellip4_75_5.res_phi[lay_gp80ellip4_75_5.theta_gt>80], density=True, histtype='step', bins=15, range=[-1, 1], label='ellip4')

#plt.hist(lay_gp80coarse_75_5.res_phi[lay_gp80coarse_75_5.theta_gt>80], density=True, histtype='step', bins=15, range=[-1, 1], label='coarse')
#plt.hist(lay_gp80infill_75_5.res_phi[lay_gp80infill_75_5.theta_gt>80], density=True, histtype='step', bins=15, range=[-1, 1], label='infill')
#plt.hist(lay_gp300_75_5.res_phi[lay_gp300_75_5.theta_gt>80], density=True, histtype='step', bins=15, range=[-1, 1], label='gp300')
plt.legend()
plt.xlabel('Azimuth PWF residues [deg]')
plt.ylabel('density')
plt.title('PWF azimuth residues for zenith >80deg')
plt.tight_layout()
plt.savefig('hist_azimuth_residues_p80.png')

plt.figure()
plt.hist(lay_gp80hexa_75_5.res_theta[lay_gp80hexa_75_5.theta_gt>80], density=True, histtype='step', bins=10, range=[-1, 1], label='hexa')
plt.hist(lay_gp80ellip_75_5.res_theta[lay_gp80ellip_75_5.theta_gt>80], density=True, histtype='step', bins=10, range=[-1, 1], label='ellip')
plt.hist(lay_gp80ellip4_75_5.res_theta[lay_gp80ellip4_75_5.theta_gt>80], density=True, histtype='step', bins=10, range=[-1, 1], label='ellip4')

#plt.hist(lay_gp80coarse_75_5.res_theta[lay_gp80coarse_75_5.theta_gt>80], density=True, histtype='step', bins=10, range=[-1, 1], label='coarse')
#plt.hist(lay_gp80infill_75_5.res_theta[lay_gp80infill_75_5.theta_gt>80], density=True, histtype='step', bins=10, range=[-1, 1], label='infill')
#plt.hist(lay_gp300_75_5.res_theta[lay_gp300_75_5.theta_gt>80], density=True, histtype='step', bins=10, range=[-1, 1], label='gp300')
plt.legend()
plt.xlabel('Zenith PWF residues [deg]')
plt.ylabel('density')
plt.title('PWF zenith residues for zenith >80deg')
plt.tight_layout()
plt.savefig('hist_zenith_residues_p80.png')

plt.figure()
plt.hist(lay_gp80hexa_75_5.res_phi[lay_gp80hexa_75_5.theta_gt<=80], density=True, histtype='step', bins=15, range=[-1, 1], label='hexa')
plt.hist(lay_gp80ellip_75_5.res_phi[lay_gp80ellip_75_5.theta_gt<=80], density=True, histtype='step', bins=15, range=[-1, 1], label='ellip')
plt.hist(lay_gp80ellip4_75_5.res_phi[lay_gp80ellip4_75_5.theta_gt<=80], density=True, histtype='step', bins=15, range=[-1, 1], label='ellip4')

#plt.hist(lay_gp80coarse_75_5.res_phi[lay_gp80coarse_75_5.theta_gt>80], density=True, histtype='step', bins=15, range=[-1, 1], label='coarse')
#plt.hist(lay_gp80infill_75_5.res_phi[lay_gp80infill_75_5.theta_gt>80], density=True, histtype='step', bins=15, range=[-1, 1], label='infill')
#plt.hist(lay_gp300_75_5.res_phi[lay_gp300_75_5.theta_gt>80], density=True, histtype='step', bins=15, range=[-1, 1], label='gp300')
plt.legend()
plt.xlabel('Azimuth PWF residues [deg]')
plt.ylabel('density')
plt.title('PWF azimuth residues for zenith <=80deg')
plt.tight_layout()
plt.savefig('hist_azimuth_residues_m80.png')

plt.figure()
plt.hist(lay_gp80hexa_75_5.res_theta[lay_gp80hexa_75_5.theta_gt<=80], density=True, histtype='step', bins=10, range=[-1, 1], label='hexa')
plt.hist(lay_gp80ellip_75_5.res_theta[lay_gp80ellip_75_5.theta_gt<=80], density=True, histtype='step', bins=10, range=[-1, 1], label='ellip')
plt.hist(lay_gp80ellip4_75_5.res_theta[lay_gp80ellip4_75_5.theta_gt<=80], density=True, histtype='step', bins=10, range=[-1, 1], label='ellip4')
#plt.hist(lay_gp80coarse_75_5.res_theta[lay_gp80coarse_75_5.theta_gt>80], density=True, histtype='step', bins=10, range=[-1, 1], label='coarse')
#plt.hist(lay_gp80infill_75_5.res_theta[lay_gp80infill_75_5.theta_gt>80], density=True, histtype='step', bins=10, range=[-1, 1], label='infill')
#plt.hist(lay_gp300_75_5.res_theta[lay_gp300_75_5.theta_gt>80], density=True, histtype='step', bins=10, range=[-1, 1], label='gp300')
plt.legend()
plt.xlabel('Zenith PWF residues [deg]')
plt.ylabel('density')
plt.title('PWF zenith residues for zenith <=80deg')
plt.tight_layout()
plt.savefig('hist_zenith_residues_m80.png')