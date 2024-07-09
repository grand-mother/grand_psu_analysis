

import numpy as np
import glob
import matplotlib.pyplot as plt

# better plots
from matplotlib import rc

import os
from grid_shape_lib.modules import masks as masks
import grand_psu_lib.modules.layout_dc2 as ldu


# rc('font', **{'family':'serif','serif':['Palatino']})
# rc('text', usetex = True)
from matplotlib import rc
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






def load_all_antennas_GP300_Paul():
    # this event contains all the 289 antennas
    ev_id = 8000
    res_file = os.path.join(data_dir_GP300_Paul, 'data_files/{}.npy'.format(ev_id))

    arr = np.load(res_file)
    du_names = arr[:, 0]
    du_pos = arr[:, 1:4]

    return du_pos, du_names

def load_all_antennas():
    # this event contains all the 289 antennas
    ev_id = 85
    res_file = os.path.join(data_dir_zhaires_nj, 'data_files/{}.npy'.format(ev_id))

    arr = np.load(res_file)
    du_names = arr[:, 0]
    du_pos = arr[:, 1:4]

    return du_pos, du_names

def load_all_antennas_gaa():
    # this event contains all the 289 antennas
    ev_id = 68
    res_file = os.path.join(data_gaa, 'data_files/{}.npy'.format(ev_id))

    arr = np.load(res_file)
    du_names = arr[:, 0]
    du_pos = arr[:, 1:4]

    return du_pos, du_names



n_trig_thres = 4


data_gaa = '/Users/ab212678/Documents/GRAND/sims/gaa_eventrate_with_tested_cores/Zhaires-NJ'

data_dir_zhaires_nj = '/Users/ab212678/Documents/GRAND/sims/DC2/DC2Training/PWFdata_3june24/Zhaires_NJ'

output_dir = '/Users/ab212678/Documents/GRAND/sims/DC2/Simu_Paul_gp300_test_grandpsuanalysis/'

data_dir_GP300_Paul = '/Users/ab212678/Documents/GRAND/sims/sims_paul/'


data_dir_GP300_Paul_all = '/Users/ab212678/Documents/GRAND/sims/outputs_simu_paul_renamed/'

gp13_id = [138, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288]

du_pos_all_gaa, du_names_all_gaa = load_all_antennas_gaa()
du_pos_all, du_names_all = load_all_antennas()
du_pos_all_gp300p, du_names_all_gp300p = load_all_antennas_GP300_Paul()


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




### load gp80 positions




### reload the .txt meters and redo the plots to make sure
gp300_xy = np.loadtxt('./gp80_positions/gp300_official_position_meters.txt', skiprows=1)
gp80_coarse = np.loadtxt('gp80_positions/gp80_coarse_position_meters.txt', skiprows=1)
gp80_infill = np.loadtxt('gp80_positions/gp80_infill_position_meters.txt', skiprows=1)
gp80_ellip = np.loadtxt('gp80_positions/gp80_elliptical_position_meters.txt', skiprows=1)
gp80_hybrid = np.loadtxt('gp80_positions/gp80_hybrid_position_meters.txt', skiprows=1)
gp13 = np.loadtxt('gp80_positions/gp13_position_meters.txt', skiprows=1)






plt.figure(35, figsize=(10, 8))
plt.clf()
plt.plot(-gp300_xy[:, 1], gp300_xy[:, 0], 'co', ms=7, alpha=0.4, label='GP300')
plt.plot(-gp80_coarse[:, 1], gp80_coarse[:, 0], 'k*', ms=10, alpha=0.4, label='GP80 coarse')
plt.plot(-gp80_ellip[:, 1], gp80_ellip[:, 0], 'bo', ms=12, fillstyle='none', label='GP80 elliptic')
plt.plot(-gp80_hybrid[:, 1], gp80_hybrid[:, 0], 'orange', ls = 'None', marker='.', ms=11, label='GP80 hybrid')
plt.plot(-gp80_infill[:, 1], gp80_infill[:, 0], 'rh', ms=2, label='GP80 infill')
plt.plot(-gp13[:, 2], gp13[:, 1], 'gh', ms=2, label='GP13')
#for i in range(10):
#    plt.text(-xy_coords.y[i], xy_coords.x[i], '{}'.format(int(coords_data[i, 0])))
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


names_gp13 = get_sublayout_names(gp13[:, 1:3], du_pos_all_gp300p, du_names_all_gp300p)
names_ellip = get_sublayout_names(gp80_ellip, du_pos_all_gp300p, du_names_all_gp300p)
names_coarse = get_sublayout_names(gp80_coarse, du_pos_all_gp300p, du_names_all_gp300p)
names_hybrid = get_sublayout_names(gp80_hybrid, du_pos_all_gp300p, du_names_all_gp300p)
names_infill = get_sublayout_names(gp80_infill, du_pos_all_gp300p, du_names_all_gp300p)


lay_gp13_75_4 = ldu.Layout_dc2(
    du_pos_all_gp300p, du_names_all_gp300p,
    data_dir_GP300_Paul_all, layout_name='gp13_75_4',
    output_dir=output_dir,
    du_names=names_gp13,
    threshold=75, n_trig_thres=4,
    do_noise_timing=True,
    sigma_timing=5e-9,
    is_coreas=True
)
lay_gp13_75_4.make_plots()


lay_gp80ellip_75_4 = ldu.Layout_dc2(
        du_pos_all_gp300p, du_names_all_gp300p,
        data_dir_GP300_Paul_all, layout_name='gp80ellip_75_4',
        output_dir=output_dir,
        du_names=names_ellip,
        threshold=75, n_trig_thres=4,
        do_noise_timing=True,
        sigma_timing=5e-9,
        is_coreas=True
)
lay_gp80ellip_75_4.make_plots()


lay_gp80hexa_75_4 = ldu.Layout_dc2(
        du_pos_all_gp300p, du_names_all_gp300p,
        data_dir_GP300_Paul_all, layout_name='gp80hexa_75_4',
        output_dir=output_dir,
        du_names=names_hybrid,
        threshold=75, n_trig_thres=4,
        do_noise_timing=True,
        sigma_timing=5e-9,
        is_coreas=True
)
lay_gp80hexa_75_4.make_plots()


lay_gp80coarse_75_4 = ldu.Layout_dc2(
    du_pos_all_gp300p, du_names_all_gp300p,
    data_dir_GP300_Paul_all, layout_name='gp80coarse_75_4',
    output_dir=output_dir,
    du_names=names_coarse,
    threshold=75, n_trig_thres=4,
    do_noise_timing=True,
    sigma_timing=5e-9,
    is_coreas=True
)
lay_gp80coarse_75_4.make_plots()


lay_gp80infill_75_4 = ldu.Layout_dc2(
    du_pos_all_gp300p, du_names_all_gp300p,
    data_dir_GP300_Paul_all, layout_name='gp80infill_75_4',
    output_dir=output_dir,
    du_names=names_infill,
    threshold=75, n_trig_thres=4,
    do_noise_timing=True,
    sigma_timing=5e-9,
    is_coreas=True
)
lay_gp80infill_75_4.make_plots()








binning1 = Binning(n_bin_energy=20, n_bin_zen=10, energy_min=15.8)

lay_gp80infill_75_4.compute_effective_area(binning1, cst_A_Sim=194.5e6)
lay_gp80infill_75_4.plot_binned_quantities(binning1)

lay_gp80coarse_75_4.compute_effective_area(binning1, cst_A_Sim=194.5e6)
lay_gp80coarse_75_4.plot_binned_quantities(binning1)


lay_gp80hexa_75_4.compute_effective_area(binning1, cst_A_Sim=195.4e6)
lay_gp80hexa_75_4.plot_binned_quantities(binning1)


lay_gp80ellip_75_4.compute_effective_area(binning1, cst_A_Sim=194.5e6)
lay_gp80ellip_75_4.plot_binned_quantities(binning1)


lay_gp13_75_4.compute_effective_area(binning1, cst_A_Sim=194.5e6)
lay_gp13_75_4.plot_binned_quantities(binning1)




lays = [
    lay_gp80infill_75_4, lay_gp80coarse_75_4,
    lay_gp80hexa_75_4, lay_gp80ellip_75_4,
]
for lay in lays:
    print('Total number of event per day for {} = {}\n'.format(lay.layout_name, lay.rate_per_day_per_m2.sum()))



# make global plot

plt.figure(figsize=(10, 8))
for lay in lays:
    plt.stairs(lay.rate_per_day_per_m2.sum(axis=0), binning1.energy_bin_edges, ls='-', lw=2, label=lay.layout_name)
plt.legend()
plt.yscale('log')
plt.ylabel('#Event per day')
plt.xlabel('log10 Energy/eV')
plt.savefig(os.path.join(output_dir, 'event_rate_gp80_vs_energy.png'))



plt.figure(figsize=(10, 8))
for lay in lays:
    plt.stairs(lay.rate_per_day_per_m2.sum(axis=1), binning1.zen_bin_edges, ls='-', lw=2, label=lay.layout_name)
plt.legend()
plt.yscale('log')
plt.ylabel('#Event per day')
plt.xlabel('Zenith [deg]')
plt.savefig(os.path.join(output_dir, 'event_rate_gp80_vs_zenith.png'))




lays = [
  #  lay_gp80infill_75_4,
  
    lay_gp80hexa_75_4, lay_gp80ellip_75_4,
]

### plots of residues histogram

plt.figure(3, figsize=(10, 8))
plt.clf()

for lay in lays:
    plt.hist(lay.res_phi, bins=20, range=[-1.5, 1.5], label=lay.layout_name, alpha=0.4, density=True)
plt.legend()

plt.ylabel('Frequency')
plt.xlabel('Azimuth residues')
plt.savefig(os.path.join(output_dir, 'hist_residues_azimuth_hexa_ellip.png'))




plt.figure(4, figsize=(10, 8))
plt.clf()

for lay in lays:
    plt.hist(lay.res_theta, bins=20, range=[-1.5, 1.5], label=lay.layout_name, alpha=0.4, density=True)
plt.legend()

plt.ylabel('Frequency')
plt.xlabel('Zenith residues')
plt.savefig(os.path.join(output_dir, 'hist_residues_zenith_hexa_ellip.png'))

lays = [
  #  lay_gp80infill_75_4,
  
    lay_gp80hexa_75_4, lay_gp80ellip_75_4,   lay_gp80coarse_75_4,
]

### plots of residues histogram

plt.figure(3, figsize=(10, 8))
plt.clf()

for lay in lays:
    plt.hist(lay.res_phi, bins=20, range=[-1.5, 1.5], label=lay.layout_name, alpha=0.4, density=True)
plt.legend()

plt.ylabel('Frequency')
plt.xlabel('Azimuth residues')
plt.savefig(os.path.join(output_dir, 'hist_residues_azimuth_hexa_ellip_coarse.png'))




plt.figure(4, figsize=(10, 8))
plt.clf()

for lay in lays:
    plt.hist(lay.res_theta, bins=20, range=[-1.5, 1.5], label=lay.layout_name, alpha=0.4, density=True)
plt.legend()

plt.ylabel('Frequency')
plt.xlabel('Zenith residues')
plt.savefig(os.path.join(output_dir, 'hist_residues_zenith_hexa_ellip_coarse.png'))





lays = [
    #lay_gp80infill_75_4,
    lay_gp80hexa_75_4, 
    lay_gp80ellip_75_4,
    #lay_gp80coarse_75_4,
]

plt.figure()
for lay in lays:
    plt.scatter(lay.n_ant, lay.res_phi, s=4, label=lay.layout_name)
plt.legend()






lay_gp80hexa_75_4.res_phi[np.abs(lay_gp80hexa_75_4.res_phi)<1.5].std()
lay_gp80ellip_75_4.res_phi[np.abs(lay_gp80ellip_75_4.res_phi)<1.5].std()


lay_gp80hexa_75_4.res_theta[np.abs(lay_gp80hexa_75_4.res_theta)<1.5].std()
lay_gp80ellip_75_4.res_theta[np.abs(lay_gp80ellip_75_4.res_theta)<1.5].std()