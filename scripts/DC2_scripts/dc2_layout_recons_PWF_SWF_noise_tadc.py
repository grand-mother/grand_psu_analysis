

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


data_pwf_swf_nn = '/Users/ab212678/Documents/GRAND/Codes/grand_psu_analysis/dc2_0000_nonoised/'
data_pwf_swf_nj = '/Users/ab212678/Documents/GRAND/Codes/grand_psu_analysis/dc2_0000_noised/'


data_dir_zhaires_nj = '/Users/ab212678/Documents/GRAND/sims/DC2/DC2Training/PWFdata_3june24/Zhaires_NJ'



output_dir = '/Users/ab212678/Documents/GRAND/Codes/grand_psu_analysis/study_pwf_noise/'



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






def get_sublayout_names(sublayout_pos, main_lay_pos, main_lay_names):
    ids = []
    for pos in sublayout_pos:
        ids.append(int(masks.get_closest_antenna(pos[0], pos[1], main_lay_pos[:, 0:2].T )))

    return list(np.int32(main_lay_names[ids]))














lay1_75_5_nn_ef = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nn, layout_name='all289_nn_75_5_ef',
    output_dir=output_dir,
    threshold=75, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef'
)
lay1_75_5_nn_ef.make_plots()



lay1_75_5_nj_ef = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nj, layout_name='all289_nj_75_5_ef',
    output_dir=output_dir,
    threshold=75, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef'
)

lay1_75_5_nj_ef.make_plots()


lay1_25_5_nj_ef = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nj, layout_name='all289_nj_25_5_ef',
    output_dir=output_dir,
    threshold=25, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef'
)

lay1_25_5_nj_ef.make_plots()


lay1_50_5_nj_ef = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nj, layout_name='all289_nj_50_5_ef',
    output_dir=output_dir,
    threshold=50, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='ef'
)

lay1_50_5_nj_ef.make_plots()




lay1_6_5_nn_tadc = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nn, layout_name='all289_nn_6_5_tadc',
    output_dir=output_dir,
    threshold=6, n_trig_thres=5,
    do_noise_timing=True,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc'
)
lay1_6_5_nn_tadc.make_plots()



lay1_6_5_nj_tadc = ldu.Layout_dc2(
    du_pos_all, du_names_all,
    data_pwf_swf_nj, layout_name='all289_nj_6_5_tadc',
    output_dir=output_dir,
    threshold=6, n_trig_thres=5,
    do_noise_timing=False,
    sigma_timing=5e-9, do_swf=False, qty_to_use='tadc'
)
lay1_6_5_nj_tadc.make_plots()


