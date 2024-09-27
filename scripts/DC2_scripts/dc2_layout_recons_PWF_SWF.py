

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




### load gp80 positions




# ### reload the .txt meters and redo the plots to make sure
# gp300_xy = np.loadtxt('./gp80_positions/gp300_official_position_meters.txt', skiprows=1)
# gp80_coarse = np.loadtxt('gp80_positions/gp80_coarse_position_meters.txt', skiprows=1)
# gp80_infill = np.loadtxt('gp80_positions/gp80_infill_position_meters.txt', skiprows=1)
# gp80_ellip = np.loadtxt('gp80_positions/gp80_elliptical_position_meters.txt', skiprows=1)
# gp80_hybrid = np.loadtxt('gp80_positions/gp80_hybrid_position_meters.txt', skiprows=1)
# gp13 = np.loadtxt('gp80_positions/gp13_position_meters.txt', skiprows=1)






# plt.figure(35, figsize=(10, 8))
# plt.clf()
# plt.plot(-gp300_xy[:, 1], gp300_xy[:, 0], 'co', ms=7, alpha=0.4, label='GP300')
# plt.plot(-gp80_coarse[:, 1], gp80_coarse[:, 0], 'k*', ms=10, alpha=0.4, label='GP80 coarse')
# plt.plot(-gp80_ellip[:, 1], gp80_ellip[:, 0], 'bo', ms=12, fillstyle='none', label='GP80 elliptic')
# plt.plot(-gp80_hybrid[:, 1], gp80_hybrid[:, 0], 'orange', ls = 'None', marker='.', ms=11, label='GP80 hybrid')
# plt.plot(-gp80_infill[:, 1], gp80_infill[:, 0], 'rh', ms=2, label='GP80 infill')
# plt.plot(-gp13[:, 2], gp13[:, 1], 'gh', ms=2, label='GP13')
# #for i in range(10):
# #    plt.text(-xy_coords.y[i], xy_coords.x[i], '{}'.format(int(coords_data[i, 0])))
# plt.xlabel('Easting [m]')
# plt.ylabel('Northing [m]')
# plt.title('GP300 and GP80 candidates positions ')
# plt.axis('equal')
# plt.legend()
# plt.tight_layout()




def get_sublayout_names(sublayout_pos, main_lay_pos, main_lay_names):
    ids = []
    for pos in sublayout_pos:
        ids.append(int(masks.get_closest_antenna(pos[0], pos[1], main_lay_pos[:, 0:2].T )))

    return list(np.int32(main_lay_names[ids]))



# names_gp13 = get_sublayout_names(gp13[:, 1:3], du_pos_all_gp300p, du_names_all_gp300p)
# names_ellip = get_sublayout_names(gp80_ellip, du_pos_all_gp300p, du_names_all_gp300p)
# names_coarse = get_sublayout_names(gp80_coarse, du_pos_all_gp300p, du_names_all_gp300p)
# names_hybrid = get_sublayout_names(gp80_hybrid, du_pos_all_gp300p, du_names_all_gp300p)
# names_infill = get_sublayout_names(gp80_infill, du_pos_all_gp300p, du_names_all_gp300p)




# lay_gp13_ingp300_all_75_4 = ldu.Layout_dc2(
#     du_pos_all_gp300p, du_names_all_gp300p,
#     data_dir_GP300_Paul, layout_name='gp13_ingp300_all_75_4',
#     output_dir=output_dir,
#     du_names=names_gp13,
#     threshold=75, n_trig_thres=4,
#     do_noise_timing=True,
#     sigma_timing=5e-9
# )
# #lay_gp13_ingp300_all_75_4.make_plots()




# lay_gp80ellip_ingp300_all_75_4_all = ldu.Layout_dc2(
#         du_pos_all_gp300p, du_names_all_gp300p,
#         data_dir_GP300_Paul_all, layout_name='gp80ellip_ingp300_all_75_4_all',
#         output_dir=output_dir,
#         du_names=names_ellip,
#         threshold=75, n_trig_thres=4,
#         do_noise_timing=True,
#         sigma_timing=5e-9
# )
# lay_gp80ellip_ingp300_all_75_4_all.make_plots()


# lay_gp80hybrid_ingp300_all_75_4_all = ldu.Layout_dc2(
#         du_pos_all_gp300p, du_names_all_gp300p,
#         data_dir_GP300_Paul_all, layout_name='gp80hybrid_ingp300_all_75_4_all',
#         output_dir=output_dir,
#         du_names=names_hybrid,
#         threshold=75, n_trig_thres=4,
#         do_noise_timing=True,
#         sigma_timing=5e-9
# )
# lay_gp80hybrid_ingp300_all_75_4_all.make_plots()




# lay_gp80ellip_ingp300_all_75_4 = ldu.Layout_dc2(
#     du_pos_all_gp300p, du_names_all_gp300p,
#     data_dir_GP300_Paul, layout_name='gp80ellip_ingp300_all_75_4',
#     output_dir=output_dir,
#     du_names=names_ellip,
#     threshold=75, n_trig_thres=4,
#     do_noise_timing=True,
#     sigma_timing=5e-9
# )
# #lay_gp80ellip_ingp300_all_75_4.make_plots()

# lay_gp80coarse_ingp300_all_75_4 = ldu.Layout_dc2(
#     du_pos_all_gp300p, du_names_all_gp300p,
#     data_dir_GP300_Paul, layout_name='gp80coarse_ingp300_all_75_4',
#     output_dir=output_dir,
#     du_names=names_coarse,
#     threshold=75, n_trig_thres=4,
#     do_noise_timing=True,
#     sigma_timing=5e-9
# )
# #lay_gp80coarse_ingp300_all_75_4.make_plots()





# lay_gp80hybrid_ingp300_all_75_4 = ldu.Layout_dc2(
#     du_pos_all_gp300p, du_names_all_gp300p,
#     data_dir_GP300_Paul, layout_name='gp80hybrid_ingp300_all_75_4',
#     output_dir=output_dir,
#     du_names=names_hybrid,
#     threshold=75, n_trig_thres=4,
#     do_noise_timing=True,
#     sigma_timing=5e-9
# )
# #lay_gp80hybrid_ingp300_all_75_4.make_plots()




# lay_gp80infill_ingp300_all_75_4 = ldu.Layout_dc2(
#     du_pos_all_gp300p, du_names_all_gp300p,
#     data_dir_GP300_Paul, layout_name='gp80infill_ingp300_all_75_4',
#     output_dir=output_dir,
#     du_names=names_infill,
#     threshold=75, n_trig_thres=4,
#     do_noise_timing=True,
#     sigma_timing=5e-9
# )
# #lay_gp80infill_ingp300_all_75_4.make_plots()








# lay_gp80ellip_ingp300_all_30_4 = ldu.Layout_dc2(
#     du_pos_all_gp300p, du_names_all_gp300p,
#     data_dir_GP300_Paul, layout_name='gp80ellip_ingp300_all_30_4',
#     output_dir=output_dir,
#     du_names=names_ellip,
#     threshold=30, n_trig_thres=4,
#     do_noise_timing=True,
#     sigma_timing=5e-9
# )
# #lay_gp80ellip_ingp300_all_30_4.make_plots()


# lay_gp80hybrid_ingp300_all_30_4 = ldu.Layout_dc2(
#     du_pos_all_gp300p, du_names_all_gp300p,
#     data_dir_GP300_Paul, layout_name='gp80hybrid_ingp300_all_30_4',
#     output_dir=output_dir,
#     du_names=names_hybrid,
#     threshold=30, n_trig_thres=4,
#     do_noise_timing=True,
#     sigma_timing=5e-9
# )
# #lay_gp80hybrid_ingp300_all_30_4.make_plots()









# lay_gp300p_all_75_4_NJ = ldu.Layout_dc2(
#     du_pos_all_gp300p, du_names_all_gp300p,
#     data_dir_GP300_Paul, layout_name='gp300p_all_NJ_75_4',
#     output_dir=output_dir,
#     threshold=75, n_trig_thres=4,
#     do_noise_timing=True,
#     sigma_timing=5e-9
# )
# #lay_gp300p_all_75_4_NJ.make_plots()


# lay1_100_6_NJ = ldu.Layout_dc2(
#     du_pos_all, du_names_all,
#     data_pwf_swf, layout_name='all289_NJ_100_6_new_v2',
#     output_dir=output_dir,
#     threshold=100, n_trig_thres=6,
#     do_noise_timing=True,
#     sigma_timing=5e-9
# )
# evt = lay1_100_6_NJ.event_res_tab



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





plt.figure(1)
plt.clf()
plt.hist(lay1_75_5_NJ.res_phi, bins=50, range=[-3, 3], histtype='step',  label='PWF')
plt.hist(lay1_75_5_NJ.res_phi_swf, bins=50, range=[-3, 3], histtype='step', label='SWF ncall400')
#plt.hist(lay1_75_5_NJ_ncall200.res_phi_swf, bins=50, range=[-3, 3], histtype='step', label='SWF ncall200')
#plt.hist(lay1_75_5_NJ_ncall100.res_phi_swf, bins =50, range=[-3, 3], alpha=0.4, label='SWF ncall100')
plt.legend()
plt.xlabel('Azimuth residues [deg]')
plt.ylabel('# events')
plt.tight_layout()
plt.savefig('comp_swf_pwf_azimuth.png')



plt.figure(2)
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


plt.figure()
plt.plot(lay1_75_5_NJ.n_ant, lay1_75_5_NJ_ncall100.res_theta_swf, 'g.', label='ncall=100')
plt.plot(lay1_75_5_NJ.n_ant, lay1_75_5_NJ_ncall200.res_theta_swf, 'b.', label='ncall=200')
plt.plot(lay1_75_5_NJ.n_ant, lay1_75_5_NJ.res_theta_swf, 'r.', label='ncall=400')
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
plt.ylabel('xmax_z [m]')
plt.tight_layout()
plt.savefig('xmax_z_vs_zenith.png')



d2d_max = np.sqrt((evt[:, 17] - evt[:, 5])**2 + (evt[:, 16] - evt[:, 4])**2)
plt.figure(3)
plt.clf()
plt.scatter(evt[:, 2], evt[:, 3], c=d2d_max)
plt.title('2d distance from sc to xmax')
plt.xlabel('zenith [deg]')
plt.ylabel('azimuth [deg]')
plt.colorbar(label='d2d_max [m]')

d2d_max = np.sqrt((evt[:, 17] - evt[:, 5])**2 + (evt[:, 16] - evt[:, 4])**2)
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


    # lay_gaa_75_3_CC = ldu.Layout_dc2(
    #     du_pos_all_gaa, du_names_all_gaa,
    #     data_gaa, layout_name='gaa_NJ_75_3_CC',
    #     output_dir=output_dir,
    #     threshold=75, n_trig_thres=3,
    #     do_noise_timing=True,
    #     sigma_timing=5e-9
    # )
    # #lay_gaa_75_3_CC.make_plots()


    # lay_gaa_75_3_nCC = ldu.Layout_dc2(
    #     du_pos_all_gaa, du_names_all_gaa,
    #     data_gaa, layout_name='gaa_NJ_75_3_nCC',
    #     threshold=75, n_trig_thres=3,
    #     do_noise_timing=True,
    #     sigma_timing=5e-9
    # )
    # #lay_gaa_75_3_nCC.make_plots()



