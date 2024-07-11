
import numpy as np
import glob
import matplotlib.pyplot as plt
# better plots
from matplotlib import rc
import os
import json


import recons_PWF as pwf
import scipy
from grid_shape_lib.utils import diff_spec as tale
from grid_shape_lib.modules import hexy
# rc('font', **{'family':'serif','serif':['Palatino']})
# rc('text', usetex = True)
rc('font', size=16.0)

D2R = np.pi / 180
R2D = 180 / np.pi

c_light = 2.997924580e8

class Layout_dc2:
    def __init__(
        self,
        du_pos_base,
        du_names_base,
        input_dir,
        output_dir='',
        layout_name='',
        du_names=None,
        threshold=75,
        n_trig_thres=4,
        n_r=1.000139,
        do_noise_timing=True,
        sigma_timing=5e-9,
        is_coreas=False
    ):
        self.du_pos_base = du_pos_base
        self.du_names_base = du_names_base
        if du_names:
            self.du_names = np.int32(du_names)
        else:
            self.du_names = np.int32(du_names_base)
        self.threshold = threshold
        self.n_trig_thres = n_trig_thres
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.is_coreas = is_coreas
        if output_dir == '':
            self.output_dir = self.input_dir
        self.files_dir = os.path.join(self.input_dir, 'data_files')
        self.plot_path = os.path.join(self.output_dir, 'plots_{}'.format(layout_name))
        os.makedirs(self.plot_path, exist_ok=True)

        event_list = glob.glob(self.files_dir + '/*.json')
        self.event_list = [os.path.splitext(os.path.basename(ev))[0] for ev in event_list]

        self.do_noise_timing = do_noise_timing
        self.sigma_timing = sigma_timing
        self.n_r = n_r
        self.layout_name = layout_name
        self.A_sim = 170 * 1000**2  # m^2

        self.event_res_tab_file = os.path.join(self.plot_path, 'event_res_tab.npy')
        if os.path.isfile(self.event_res_tab_file):
            self.event_res_tab = np.load(self.event_res_tab_file)
        else:
            self.get_global_lists()
            np.save(self.event_res_tab_file, self.event_res_tab)

        self.make_shortcuts()

    def get_global_lists(self):

        global_pos_list = []
        np.random.seed(seed=6578)

        event_res_tab = np.zeros((len(self.event_list), 16)) - 1

        for k, ev_id in enumerate(self.event_list[0:]):

            res_file = os.path.join(self.files_dir, '{}.npy'.format(ev_id))
            json_file = os.path.join(self.files_dir, '{}.json'.format(ev_id))

            arr = np.load(res_file)

            with open(json_file, 'r') as f:
                event_params = json.load(f)

            event_res_tab[k, 0] = np.int32(event_params['event_number'])
            event_res_tab[k, 1] = np.log10(np.float32(event_params['energy'])*1e9)
            if self.is_coreas:
                event_res_tab[k, 1] -= 9
            event_res_tab[k, 2] = np.float32(event_params['zenith'])
            event_res_tab[k, 3] = np.float32(event_params['azimuth'])
            event_res_tab[k, 4] = np.float32(event_params['shower_core_x'])
            event_res_tab[k, 5] = np.float32(event_params['shower_core_y'])
            event_res_tab[k, 6] = np.float32(event_params['xmax_grams'])
            event_res_tab[k, 7] = np.float32(event_params['event_weight'])

            ev_du_ids = arr[:, 0]

            mask = np.array([i in self.du_names for i in ev_du_ids])

            ev_du_ids = ev_du_ids[mask]
            arr = arr[mask]

            n_du = len(ev_du_ids)

            event_res_tab[k, 8] = n_du

            if n_du > 0:
                ev_du_pos = arr[:, 1:4]

                tmax_3d = arr[:, 4:7] * 1e-9 + np.random.randn(*(arr[:, 4:7].shape)) * self.sigma_timing * self.do_noise_timing
                Emax_3d = arr[:, 7:10]

                ll = list(np.vstack([np.arange(n_du), np.argmax(Emax_3d, axis=1)]).T)
                ll = [tuple(l_) for l_ in ll]
                tmax = np.array([tmax_3d[l_] for l_ in ll])
                Emax = np.array([Emax_3d[l_] for l_ in ll])

                id_above_threshold = np.where(Emax > self.threshold)[0]
                n_above_threshold = len(id_above_threshold)
                event_res_tab[k, 9] = n_above_threshold

                if n_above_threshold >= self.n_trig_thres:

                    x_ants = ev_du_pos[id_above_threshold]
                    t_ants = tmax[id_above_threshold]

                    theta = np.float32(event_params['zenith'])
                    phi = np.float32(event_params['azimuth'])

                    theta_pred, phi_pred = pwf.PWF_semianalytical(x_ants, t_ants, cr=self.n_r, c=c_light)
                    if np.abs(phi_pred - phi*D2R) > 320 * D2R:
                        phi_pred += -360 * D2R

                    cov = pwf.Covariance_tangentplane(theta*D2R, phi*D2R, x_ants, self.sigma_timing, c=c_light, cr=self.n_r)
                    sigma_theta = np.sqrt(cov[0, 0])
                    sigma_phi = np.sqrt(cov[1, 1])

                    cov2 = pwf.fisher_Variance(theta*D2R, phi*D2R, x_ants, self.sigma_timing, c=c_light, cr=self.n_r)
                    sigma_thetaF = np.sqrt(cov2[0, 0])
                    sigma_phiF = np.sqrt(cov2[1, 1])
                    event_res_tab[k, 10] = theta_pred*R2D
                    event_res_tab[k, 11] = phi_pred*R2D
                    event_res_tab[k, 12] = sigma_theta*R2D
                    event_res_tab[k, 13] = sigma_phi*R2D

                    event_res_tab[k, 14] = sigma_thetaF*R2D
                    event_res_tab[k, 15] = sigma_phiF*R2D

                    global_pos_list.append([x_ants, t_ants])
                else:
                    global_pos_list.append(-1)
            else:
                global_pos_list.append(-1)

            self.global_pos_list = global_pos_list
            self.event_res_tab = event_res_tab
            

    def make_shortcuts(self):


        good_ids = np.where(
            (self.event_res_tab[:, 9] >= self.n_trig_thres) *
            (np.isfinite(self.event_res_tab[:, 10]))
        )[0]
        self.event_number = self.event_res_tab[good_ids, 0]

        self.theta_gt = self.event_res_tab[good_ids, 2]
        self.phi_gt = self.event_res_tab[good_ids, 3]
        self.energy = self.event_res_tab[good_ids, 1]

        self.theta_pred = self.event_res_tab[good_ids, 10]
        self.phi_pred = self.event_res_tab[good_ids, 11]

        self.sig_theta_gt = self.event_res_tab[good_ids, 14]
        self.sig_phi_gt = self.event_res_tab[good_ids, 13]

        self.sig_theta_gtF = self.event_res_tab[good_ids, 14]
        self.sig_phi_gtF = self.event_res_tab[good_ids, 15]

        self.n_ant = self.event_res_tab[good_ids, 9]
        self.shc_x = self.event_res_tab[good_ids, 4]
        self.shc_y = self.event_res_tab[good_ids, 5]
        self.d = np.sqrt(self.shc_x**2 + self.shc_y**2)

        self.res_phi = self.phi_pred - self.phi_gt
        self.res_theta = self.theta_pred - self.theta_gt

        self.id_good = np.where((np.abs(self.res_phi) < 5)*(np.abs(self.res_theta) < 5))


        

    def plot_event(self, id):

        ev_id = self.event_number[id]
        res_file = os.path.join(self.files_dir, '{}.npy'.format(int(ev_id)))
        arr = np.load(res_file)
        ev_du_ids = arr[:, 0]

        mask = np.array([i in self.du_names for i in ev_du_ids])

        ev_du_ids = ev_du_ids[mask]
        arr = arr[mask]
        n_du = len(ev_du_ids)
        ev_du_pos = arr[:, 1:4]

        tmax_3d = arr[:, 4:7] * 1e-9 + np.random.randn(*(arr[:, 4:7].shape)) * self.sigma_timing * self.do_noise_timing
        Emax_3d = arr[:, 7:10]

        ll = list(np.vstack([np.arange(n_du), np.argmax(Emax_3d, axis=1)]).T)
        ll = [tuple(l_) for l_ in ll]
        tmax = np.array([tmax_3d[l_] for l_ in ll])
        Emax = np.array([Emax_3d[l_] for l_ in ll])

        id_above_threshold = np.where(Emax > self.threshold)[0]

        x_ants = ev_du_pos[id_above_threshold]
        t_ants = tmax[id_above_threshold]


        idx = ([np.where(self.du_names_base == ib)[0] for ib in self.du_names])
        idx = [idxx[0] for idxx in idx if len(idxx) > 0]

        fig = plt.figure(figsize=(10, 6))

        plt.plot(-self.du_pos_base[:, 1]/1000, self.du_pos_base[:, 0]/1000, 'k.', alpha=0.2)
        plt.plot(-self.du_pos_base[idx, 1]/1000, self.du_pos_base[idx, 0]/1000, 'k.')
        plt.plot(-self.shc_y[id]/1000, self.shc_x[id]/1000, 'ro')

        az = self.phi_gt[id]
        plt.arrow(-self.shc_y[id]/1000, self.shc_x[id]/1000, 1 * -np.sin(np.pi/180*az+np.pi), 1*np.cos(np.pi/180*az+np.pi), zorder=10, width=.1)
        plt.arrow(-self.shc_y[id]/1000, self.shc_x[id]/1000, 5 * -np.sin(np.pi/180*az+np.pi), 5*np.cos(np.pi/180*az+np.pi), zorder=10, width=.1)
        plt.arrow(-self.shc_y[id]/1000, self.shc_x[id]/1000, 8 * -np.sin(np.pi/180*az+np.pi), 8*np.cos(np.pi/180*az+np.pi), zorder=10, width=.1)


        plt.scatter(-x_ants[:, 1]/1000, x_ants[:, 0]/1000, c=t_ants*1e6, zorder=8)
        plt.colorbar(label='tmax[ms]')
        plt.axis('equal')
        plt.xlabel('Easting [km]')
        plt.ylabel('Northing [km]')
        plt.title('Ev. nu. {}, E={:.3}, Az={:.3}, Zen={:.3}'.format(int(ev_id), self.energy[id], az, self.theta_gt[id])) 
        plt.tight_layout()
        return fig



    def make_plots(self):
        # self.plot_raw_residues()
        # self.plot_2d_residues()
        # self.plot_histogram_raw_residues()
        # self.plot_histrogram_normalized_residues()
        # self.plot_histrogram_normalized_residues_distane_cut()
        # self.plot_residues_with_error_bars()
        # self.plot_residues_with_error_bars_zoom()
        # self.plot_scatter_uncertainties()
        self.plot_events_cores()
        self.plot_layout()

    def plot_2d_residues(self):

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sc = ax.scatter(self.res_phi, self.res_theta, c=self.n_ant, vmin=3, vmax=10, cmap='Paired')

        ax.set_xlabel('phi_pred - phi_gt [deg]')
        ax.set_ylabel('theta_pred - theta_gt [deg]')
        ax.set_title('2d residues')
        fig.colorbar(sc, label='# antennas')
        fig.tight_layout()

        plt.savefig(os.path.join(self.plot_path, '2d_residues_{}.png'.format(self.layout_name)))

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sc = ax.scatter(self.res_phi, self.res_theta, c=self.n_ant, s=4, vmin=3, vmax=10, cmap='Paired')

        ax.set_xlabel('phi_pred - phi_gt [deg]')
        ax.set_ylabel('theta_pred - theta_gt [deg]')
        ax.set_title('2d residues')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        fig.colorbar(sc, label='# antennas')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, '2d_residues_zoom_{}.png'.format(self.layout_name)))

    def plot_layout(self):

       

        idx = ([np.where(self.du_names_base == ib)[0] for ib in self.du_names])
        idx = [idxx[0] for idxx in idx if len(idxx) > 0]


        #idx = np.where(self.du_ids == self.du_names_base)[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(-self.du_pos_base[:, 1]/1000, self.du_pos_base[:, 0]/1000, c='k', s=2, alpha=0.2)
        ax.scatter(-self.du_pos_base[idx, 1]/1000, self.du_pos_base[idx, 0]/1000, c='k', s=8, zorder=8, label=self.layout_name)

        ax.legend()
        ax.set_xlabel('Easting [km]')
        ax.set_ylabel('Northing [km]')
        ax.set_title('Position of layout antennas')
        ax.axis('equal')
        fig.tight_layout()

        plt.savefig(os.path.join(self.plot_path, 'layout_{}.png'.format(self.layout_name)))

        coords = {}
        coords["du_names_base"] = [int(idu) for idu in self.du_names_base]
        coords["du_names_layout"] = [int(idu) for idu in self.du_names]
        coords["northing"] = list(self.du_pos_base[:, 0])
        coords["westing"] = list(self.du_pos_base[:, 1])
        coords["alt"] = list(self.du_pos_base[:, 2])

        coords_json_name = os.path.join(self.plot_path, 'coords{}.json'.format(self.layout_name))
        with open(coords_json_name, 'w') as f:
            json.dump(coords, f)


    def plot_events_cores(self):
        import matplotlib.patches as mpatches
        r1=12500*(1/np.cos(np.radians(87)))/1000 /3
        r2=12500*(1/np.cos(np.radians(88)))/1000 /3
        hexagon1 = mpatches.RegularPolygon((0, 0), numVertices=6, radius=r1, orientation=np.pi / 6, edgecolor='r', facecolor='none', label='87deg')
        hexagon2 = mpatches.RegularPolygon((0, 0), numVertices=6, radius=r2, orientation=np.pi / 6, edgecolor='r', facecolor='none', label='88deg')

        hexagon3 = mpatches.RegularPolygon((0, 0), numVertices=6, radius=r1, orientation=np.pi / 6, edgecolor='r', facecolor='none', label='87deg')
        hexagon4 = mpatches.RegularPolygon((0, 0), numVertices=6, radius=r2, orientation=np.pi / 6, edgecolor='r', facecolor='none', label='88deg')

        hexagon5 = mpatches.RegularPolygon((0, 0), numVertices=6, radius=r1, orientation=np.pi / 6, edgecolor='r', facecolor='none', label='87deg')
        hexagon6 = mpatches.RegularPolygon((0, 0), numVertices=6, radius=r2, orientation=np.pi / 6, edgecolor='r', facecolor='none', label='88deg')


        idx = ([np.where(self.du_names_base == ib)[0] for ib in self.du_names])
        idx = [idxx[0] for idxx in idx if len(idxx)>0]

        fig, ax = plt.subplots(1, 1)
        ax.scatter(-self.du_pos_base[:, 1]/1000, self.du_pos_base[:, 0]/1000, c='k', s=2, alpha=0.2)
        ax.scatter(-self.du_pos_base[idx, 1]/1000, self.du_pos_base[idx, 0]/1000, c='k', s=8, zorder=8)
        sc = ax.scatter(-self.shc_y/1000, self.shc_x/1000, c=self.phi_gt, cmap='jet', s=4)
        #ax.add_patch(hexagon1)
        #ax.add_patch(hexagon2)
        ax.legend()
        ax.set_xlabel('Easting [km]')
        ax.set_ylabel('Northing [km]')
        fig.colorbar(sc, label='Azimuth [deg]')
        ax.set_title('Position of triggered cores\n {}'.format(self.layout_name))
        ax.axis('equal')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'triggered_cores_vs_azimuth_{}.png'.format(self.layout_name)))

        fig, ax = plt.subplots(1, 1)
        ax.scatter(-self.du_pos_base[:, 1]/1000, self.du_pos_base[:, 0]/1000, c='k', s=2, alpha=0.2)
        ax.scatter(-self.du_pos_base[idx, 1]/1000, self.du_pos_base[idx, 0]/1000, c='k', s=8, zorder=8)
        sc = ax.scatter(-self.shc_y/1000, self.shc_x/1000, c=self.energy, cmap='jet', s=4)
        #ax.add_patch(hexagon3)
        #ax.add_patch(hexagon4)
        ax.legend()
        ax.set_xlabel('Easting [km]')
        ax.set_ylabel('Northing [km]')
        ax.set_title('Position of triggered cores\n {}'.format(self.layout_name))
        ax.axis('equal')
        fig.colorbar(sc, label='log10 Energy/eV ')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'triggered_cores_vs_energy_{}.png'.format(self.layout_name)))

        fig, ax = plt.subplots(1, 1)
        ax.scatter(-self.du_pos_base[:, 1]/1000, self.du_pos_base[:, 0]/1000, c='k', s=2, alpha=0.2)
        ax.scatter(-self.du_pos_base[idx, 1]/1000, self.du_pos_base[idx, 0]/1000, c='k', s=8, zorder=8)
        sc = ax.scatter(-self.shc_y/1000, self.shc_x/1000, c=self.theta_gt, cmap='jet', s=4)
        #ax.add_patch(hexagon5)
        #ax.add_patch(hexagon6)
        ax.legend()
        ax.set_xlabel('Easting [km]')
        ax.set_ylabel('Northing [km]')
        ax.set_title('Position of triggered cores\n {}'.format(self.layout_name))
        ax.axis('equal')
        fig.colorbar(sc, label='Zenith [deg]')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'triggered_cores_vs_zenith_{}.png'.format(self.layout_name)))

    def plot_raw_residues(self):

        plt.figure(3)
        plt.clf()
        plt.plot(self.theta_gt, self.theta_pred - self.theta_gt, 'k.')
        plt.xlabel('theta_gt')
        plt.ylabel('theta_pred - theta_gt')
        plt.title('zenith residual vs zenith')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'zen_residues_vs_zenith_v1_{}.png'.format(self.layout_name)))

        plt.figure(4)
        plt.clf()
        plt.plot(self.phi_gt, self.theta_pred - self.theta_gt, 'k.')
        plt.xlabel('phi_gt')
        plt.ylabel('theta_pred - theta_gt')
        plt.title('zenith residual vs azimuth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'zen_residues_vs_azimuth_v1_{}.png'.format(self.layout_name)))

        plt.figure(5)
        plt.clf()
        plt.plot(self.theta_gt, self.phi_pred - self.phi_gt, 'k.')
        plt.xlabel('theta_gt')
        plt.ylabel('phi_pred - phi_gt')
        plt.title('azimuth residual vs zenith')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'az_residues_vs_zenith_v1_{}.png'.format(self.layout_name)))

        plt.figure(6)
        plt.clf()
        plt.plot(self.phi_gt, self.phi_pred - self.phi_gt, 'k.')
        plt.xlabel('phi_gt')
        plt.ylabel('phi_pred - phi_gt')
        plt.title('azimuth residual vs azimuth')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'az_residues_vs_azimuth_v1_{}.png'.format(self.layout_name)))

    def plot_histogram_raw_residues(self):
        # use only the event where the residual are below 10deg in az or in zen
        id_phi = np.where(np.abs(self.phi_pred - self.phi_gt)< 10)[0]

        plt.figure(111)
        plt.clf()
        h = plt.hist((self.phi_pred - self.phi_gt)[id_phi], bins=50, range=[-2, 2], density=True)
        plt.text(-2, 2.5, '< phi_pred-phi_gt> = {:.3} deg'.format((self.phi_pred - self.phi_gt)[id_phi].mean()))
        plt.text(-2, 2.2, 'RMS(phi_pred - phi_gt) = {:.3} deg'.format((self.phi_pred - self.phi_gt)[id_phi].std()))
        #plt.plot(h[1], scipy.stats.norm.pdf(h[1], (phi_pred_NJ - phi_gt_NJ).mean(), (phi_pred_NJ - phi_gt_NJ).std()), label='N(0, 1)')
        plt.legend(loc=0)
        plt.xlabel('phi_pred - phi_gt')
        plt.ylabel('frequency')
        plt.title('Azimuth residues')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'hist_azimuth_residues_v1_{}.png'.format(self.layout_name)))

        id_theta = np.where(np.abs(self.theta_pred - self.theta_gt)< 10)[0]
        plt.figure(211)
        plt.clf()
        h = plt.hist((self.theta_pred - self.theta_gt)[id_theta], bins=50, range=[-2, 2], density=True)
        plt.text(-2, 2.5, '< theta_pred-theta_gt> = {:.3} deg'.format((self.theta_pred - self.theta_gt)[id_theta].mean()))
        plt.text(-2, 2.2, 'RMS(theta_pred - theta_gt) = {:.3} deg'.format((self.theta_pred - self.theta_gt)[id_theta].std()))
        plt.legend(loc=0)
        plt.xlabel('theta_pred - theta_gt')
        plt.ylabel('frequency')
        plt.title('zenith residues')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'hist_zenith_residues_DC2_v1_{}.png'.format(self.layout_name)))


        # binning in zenith
        n_bins = 10
        zen_bin_edges = np.linspace(60, 88, n_bins+1)
        # zen_bin_centers = (zen_bin_edges[1:] + zen_bin_edges[:-1]) * 0.5 

        bin_indices = np.digitize(self.theta_gt, zen_bin_edges)

        for i in range(n_bins):
            residues_theta = self.res_theta[(bin_indices == i+1)]
            plt.figure()
            plt.clf()
            h = plt.hist(residues_theta, bins=50, range=[-2, 2], density=True, alpha=0.5)
            plt.xlabel('(theta_pred - theta_gt)')
            plt.ylabel('frequency')
            plt.title('theta residues {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'hist_zenith_residues_DC2_v1_{:.3}_zen_{:.3}.png').format(zen_bin_edges[i], zen_bin_edges[i+1]))

        for i in range(n_bins):
            residues_phi = self.res_phi[(bin_indices == i+1)]
            plt.figure()
            plt.clf()
            h = plt.hist(residues_phi, bins=50, range=[-2, 2], density=True, alpha=0.5)
            plt.xlabel('(phi_pred - phi_gt)')
            plt.ylabel('frequency')
            plt.title('phi residues {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'hist_azimuth_residues_DC2_v1_{:.3}_zen_{:.3}.png').format(zen_bin_edges[i], zen_bin_edges[i+1]))




    def plot_histrogram_normalized_residues(self):

        plt.figure(11)
        plt.clf()
        h = plt.hist((self.phi_pred - self.phi_gt)/self.sig_phi_gt, bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=1')
        plt.hist((self.phi_pred - self.phi_gt)/(2*self.sig_phi_gt), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=2')
        plt.hist((self.phi_pred - self.phi_gt)/(3*self.sig_phi_gt), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=3')
        #plt.text(-10, 0.4, '<sig_phi_gt> = {:.3} deg'.format(sig_phi_gt_NJ.mean()))
        plt.plot(h[1], scipy.stats.norm.pdf(h[1], 0, 1), label='N(0, 1)')
        plt.legend(loc=0)
        plt.xlabel('(phi_pred - phi_gt) / n sig_phi_gt')
        plt.ylabel('frequency')
        plt.title('Normalized phi residues')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'hist_azimuth_normalized_residues_DC2_v1_{}.png'.format(self.layout_name)))


        plt.figure(21)
        plt.clf()
        h = plt.hist((self.theta_pred - self.theta_gt)/self.sig_theta_gt, bins=50, range=[-10, 10], density=True, alpha=0.5,label='n=1')
        plt.hist((self.theta_pred - self.theta_gt)/(2*self.sig_theta_gt), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=2')
        plt.hist((self.theta_pred - self.theta_gt)/(3*self.sig_theta_gt), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=3')
        #plt.text(-10, 0.4, '<sig_theta_gt> = {:.3} deg'.format(sig_theta_gt_NJ.mean()))
        plt.plot(h[1], scipy.stats.norm.pdf(h[1], 0, 1))
        plt.legend(loc=0)
        plt.xlabel('(theta_pred - theta_gt) / n sig_theta_gt')
        plt.ylabel('frequency')
        plt.title('Normalized theta residues')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'hist_zenith_normalized_residues_DC2_v1_{}.png'.format(self.layout_name)))

        # binning in zenith
        n_bins = 10
        zen_bin_edges = np.linspace(60, 88, n_bins+1)
        # zen_bin_centers = (zen_bin_edges[1:] + zen_bin_edges[:-1]) * 0.5 

        bin_indices = np.digitize(self.theta_gt, zen_bin_edges)

        for i in range(n_bins):
            residues_theta = self.res_theta[bin_indices == i+1]
            uncer_theta = self.sig_theta_gt[bin_indices == i+1]

            plt.figure()
            plt.clf()
            h = plt.hist(residues_theta/uncer_theta, bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=1')
            plt.hist(residues_theta/(2*uncer_theta), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=2')
            plt.hist(residues_theta/(3*uncer_theta), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=3')
            plt.plot(h[1], scipy.stats.norm.pdf(h[1], 0, 1))
            plt.legend(loc=0)
            plt.xlabel('(theta_pred - theta_gt) / n sig_theta_gt')
            plt.ylabel('frequency')
            plt.title('Normalized theta residues {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'hist_zenith_normalized_residues_DC2_v1_{:.3}_zen_{:.3}.png').format(zen_bin_edges[i], zen_bin_edges[i+1]))

            residues_phi = self.res_phi[bin_indices == i+1]
            uncer_phi = self.sig_phi_gt[bin_indices == i+1]

            plt.figure()
            plt.clf()
            h = plt.hist(residues_phi/uncer_phi, bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=1')
            plt.hist(residues_phi/(2*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=2')
            plt.hist(residues_phi/(3*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=3')
            if i >= 7:
                plt.hist(residues_phi/(4*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=4')
                plt.hist(residues_phi/(5*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=5')
            if i == 9:
                plt.hist(residues_phi/(6*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=6')
            #plt.hist(residues_phi/((np.sqrt(1.0/np.cos(np.pi/180*zen_bin_centers[i])))*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=')
            plt.plot(h[1], scipy.stats.norm.pdf(h[1], 0, 1))
            plt.legend(loc=0)
            plt.xlabel('(phi_pred - phi_gt) / n sig_phi_gt')
            plt.ylabel('frequency')
            plt.title('Normalized phi residues {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'hist_azimuth_normalized_residues_DC2_v1_{:.3}_zen_{:.3}.png').format(zen_bin_edges[i], zen_bin_edges[i+1]))


    def plot_histrogram_normalized_residues_distane_cut(self):

        # binning in zenith
        n_bins = 10
        zen_bin_edges = np.linspace(60, 88, n_bins+1)
        # zen_bin_centers = (zen_bin_edges[1:] + zen_bin_edges[:-1]) * 0.5 

        bin_indices = np.digitize(self.theta_gt, zen_bin_edges)

        for i in range(n_bins):
            residues_theta = self.res_theta[(bin_indices == i+1)*(self.d < 5500)]
            uncer_theta = self.sig_theta_gt[(bin_indices == i+1)*(self.d < 5500)]

            plt.figure()
            plt.clf()
            h = plt.hist(residues_theta/uncer_theta, bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=1')
            plt.hist(residues_theta/(2*uncer_theta), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=2')
            plt.hist(residues_theta/(3*uncer_theta), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=3')
            plt.plot(h[1], scipy.stats.norm.pdf(h[1], 0, 1))
            plt.legend(loc=0)
            plt.xlabel('(theta_pred - theta_gt) / n sig_theta_gt')
            plt.ylabel('frequency')
            plt.title('Normalized theta residues close_to_center {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'hist_zenith_normalized_residues_DC2_v1_close_to_center_{:.3}_zen_{:.3}.png').format(zen_bin_edges[i], zen_bin_edges[i+1]))

            residues_phi = self.res_phi[(bin_indices == i+1)*(self.d < 5500)]
            uncer_phi = self.sig_phi_gt[(bin_indices == i+1)*(self.d < 5500)]

            plt.figure()
            plt.clf()
            h = plt.hist(residues_phi/uncer_phi, bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=1')
            plt.hist(residues_phi/(2*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=2')
            plt.hist(residues_phi/(3*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=3')
            #if i >= 7:
            #    plt.hist(residues_phi/(4*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=4')
            #    plt.hist(residues_phi/(5*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=5')
            #if i == 9:
            #    plt.hist(residues_phi/(6*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=6')
            #plt.hist(residues_phi/((np.sqrt(1.0/np.cos(np.pi/180*zen_bin_centers[i])))*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=')
            plt.plot(h[1], scipy.stats.norm.pdf(h[1], 0, 1))
            plt.legend(loc=0)
            plt.xlabel('(phi_pred - phi_gt) / n sig_phi_gt')
            plt.ylabel('frequency')
            plt.title('Normalized phi residues close_to_center {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'hist_azimuth_normalized_residues_DC2_v1_close_to_center_{:.3}_zen_{:.3}.png').format(zen_bin_edges[i], zen_bin_edges[i+1]))


        for i in range(n_bins):
            residues_theta = self.res_theta[(bin_indices == i+1)*(self.d > 5500)]
            uncer_theta = self.sig_theta_gt[(bin_indices == i+1)*(self.d > 5500)]

            plt.figure()
            plt.clf()
            h = plt.hist(residues_theta/uncer_theta, bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=1')
            plt.hist(residues_theta/(2*uncer_theta), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=2')
            plt.hist(residues_theta/(3*uncer_theta), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=3')
            plt.plot(h[1], scipy.stats.norm.pdf(h[1], 0, 1))
            plt.legend(loc=0)
            plt.xlabel('(theta_pred - theta_gt) / n sig_theta_gt')
            plt.ylabel('frequency')
            plt.title('Normalized theta residues far_from_center {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'hist_zenith_normalized_residues_DC2_v1_far_from_center_{:.3}_zen_{:.3}.png').format(zen_bin_edges[i], zen_bin_edges[i+1]))

            residues_phi = self.res_phi[(bin_indices == i+1)*(self.d>5500)]
            uncer_phi = self.sig_phi_gt[(bin_indices == i+1)*(self.d>5500)]

            plt.figure()
            plt.clf()
            h = plt.hist(residues_phi/uncer_phi, bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=1')
            plt.hist(residues_phi/(2*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=2')
            plt.hist(residues_phi/(3*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=3')
            if i >= 7:
                plt.hist(residues_phi/(4*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=4')
                plt.hist(residues_phi/(5*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=5')
            if i == 9:
                plt.hist(residues_phi/(6*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=6')
            #plt.hist(residues_phi/((np.sqrt(1.0/np.cos(np.pi/180*zen_bin_centers[i])))*uncer_phi), bins=50, range=[-10, 10], density=True, alpha=0.5, label='n=')
            plt.plot(h[1], scipy.stats.norm.pdf(h[1], 0, 1))
            plt.legend(loc=0)
            plt.xlabel('(phi_pred - phi_gt) / n sig_phi_gt')
            plt.ylabel('frequency')
            plt.title('Normalized phi residues far_from_center {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'hist_azimuth_normalized_residues_DC2_v1_far_from_center_{:.3}_zen_{:.3}.png').format(zen_bin_edges[i], zen_bin_edges[i+1]))




    def plot_residues_with_error_bars(self):

        ###### residues with errors bars  color coded with #ant

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        ax.errorbar(self.phi_gt, (self.phi_pred - self.phi_gt), yerr=3*self.sig_phi_gt, fmt='.',  linestyle="none", ms=1, zorder=-1)
        ax.plot(self.phi_gt, self.phi_gt-self.phi_gt, 'k-')
        sc = ax.scatter(self.phi_gt, (self.phi_pred - self.phi_gt), c=self.n_ant, s=10 )
        fig.colorbar(sc, label='# antennas')
        ax.set_title('Azimuth residues')
        ax.set_xlabel('phi_gt')
        ax.set_ylabel('phi_pred - phi_gt')
        #ax.set_ylim((-1, 1))
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'az_residues_vs_azimuth_DC2_v2_{}.png'.format(self.layout_name)))

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        ax.errorbar(self.phi_gt, (self.theta_pred - self.theta_gt), yerr=2*self.sig_theta_gt, fmt='.',  linestyle="none", ms=1, zorder=-1)
        ax.plot(self.phi_gt, self.theta_gt-self.theta_gt, 'k-')
        sc = ax.scatter(self.phi_gt, (self.theta_pred - self.theta_gt), c=self.n_ant, s=10 )
        fig.colorbar(sc, label='# antennas')
        ax.set_title('Zenith residues')
        ax.set_xlabel('phi_gt')
        ax.set_ylabel('theta_pred - theta_gt')
        #ax.set_ylim((-1, 1))
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'zen_residues_vs_azimuth_DC2_v2_{}.png'.format(self.layout_name)))

    def plot_residues_with_error_bars_zoom(self):

        ###### residues with errors bars  color coded with #ant

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        ax.errorbar(self.phi_gt, (self.phi_pred - self.phi_gt), yerr=3*self.sig_phi_gt, fmt='.',  linestyle="none", ms=1, zorder=-1)
        ax.plot(self.phi_gt, self.phi_gt-self.phi_gt, 'k-')
        sc = ax.scatter(self.phi_gt, (self.phi_pred - self.phi_gt), c=self.n_ant, s=10 )
        fig.colorbar(sc, label='# antennas')
        ax.set_title('Azimuth residues')
        ax.set_xlabel('phi_gt')
        ax.set_ylabel('phi_pred - phi_gt')
        ax.set_ylim((-3, 3))
        #ax.set_ylim((-1, 1))
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'az_residues_vs_azimuth_DC2_v3_{}.png'.format(self.layout_name)))
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        ax.errorbar(self.phi_gt, (self.theta_pred - self.theta_gt), yerr=2*self.sig_theta_gt, fmt='.',  linestyle="none", ms=1, zorder=-1)
        ax.plot(self.phi_gt, self.theta_gt-self.theta_gt, 'k-')
        sc = ax.scatter(self.phi_gt, (self.theta_pred - self.theta_gt), c=self.n_ant, s=10 )
        fig.colorbar(sc, label='# antennas')
        ax.set_title('Zenith residues')
        ax.set_xlabel('phi_gt')
        ax.set_ylabel('theta_pred - theta_gt')
        ax.set_ylim((-3, 3))
        #ax.set_ylim((-1, 1))
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'zen_residues_vs_azimuth_DC2_v3_{}.png'.format(self.layout_name)))


        # same plots but binned in enith

        ## binning in zenith
        n_bins = 10
        zen_bin_edges = np.linspace(60, 88, n_bins+1)
        zen_bin_centers = (zen_bin_edges[1:] + zen_bin_edges[:-1]) * 0.5 

        bin_indices = np.digitize(self.theta_gt, zen_bin_edges)
        bin_value = []
        bin_sigma = []

        for i in range(n_bins):
            residues_theta = (self.theta_pred - self.theta_gt)[bin_indices == i+1]
            uncer_theta = self.sig_theta_gt[bin_indices == i+1]


            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            ax.errorbar(self.phi_gt[bin_indices == i+1], (self.phi_pred - self.phi_gt)[bin_indices == i+1], yerr=3*self.sig_phi_gt[bin_indices == i+1], fmt='.',  linestyle="none", ms=1, zorder=-1)
            ax.plot(self.phi_gt[bin_indices == i+1], (self.phi_gt-self.phi_gt)[bin_indices == i+1], 'k-')
            sc = ax.scatter(self.phi_gt[bin_indices == i+1], (self.phi_pred - self.phi_gt)[bin_indices == i+1], c=self.n_ant[bin_indices == i+1], s=10 )
            fig.colorbar(sc, label='# antennas')
            ax.set_title('Azimuth residues {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            
            ax.set_xlabel('phi_gt')
            ax.set_ylabel('phi_pred - phi_gt')

            ax.set_ylim((-1, 1))
            if 'GP13' in self.layout_name:
                ax.set_ylim((-3, 3))
            fig.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'az_residues_vs_azimuth_DC2_v3_{:.3}_zen_{:.3}_{}.png').format(zen_bin_edges[i], zen_bin_edges[i+1], self.layout_name))

            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            ax.errorbar(self.phi_gt[bin_indices == i+1], (self.theta_pred - self.theta_gt)[bin_indices == i+1], yerr=2*self.sig_theta_gt[bin_indices == i+1], fmt='.',  linestyle="none", ms=1, zorder=-1)
            ax.plot(self.phi_gt[bin_indices == i+1], self.theta_gt[bin_indices == i+1]-self.theta_gt[bin_indices == i+1], 'k-')
            sc = ax.scatter(self.phi_gt[bin_indices == i+1], (self.theta_pred - self.theta_gt)[bin_indices == i+1], c=self.n_ant[bin_indices == i+1], s=10 )
            fig.colorbar(sc, label='# antennas')
            ax.set_title('Zenith residues {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))
            ax.set_xlabel('phi_gt')
            ax.set_ylabel('theta_pred - theta_gt')
            ax.set_ylim((-1, 1))
            if 'GP13' in self.layout_name:
                ax.set_ylim((-3, 3))
           
            fig.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'zen_residues_vs_azimuth_DC2_v3_{:.3}_zen_{:.3}_{}.png').format(zen_bin_edges[i], zen_bin_edges[i+1], self.layout_name))


            fig, ax = plt.subplots(1, 1, figsize=(15, 6))
            ax.errorbar(self.phi_gt[bin_indices == i+1], (self.phi_pred - self.phi_gt)[bin_indices == i+1], yerr=3*self.sig_phi_gt[bin_indices == i+1], fmt='.',  linestyle="none", ms=1, zorder=-1)
            ax.plot(self.phi_gt[bin_indices == i+1], (self.phi_gt-self.phi_gt)[bin_indices == i+1], 'k-')

            sc = ax.scatter(self.phi_gt[bin_indices == i+1], (self.phi_pred - self.phi_gt)[bin_indices == i+1], c=self.d[bin_indices == i+1], s=10 )
            fig.colorbar(sc, label='distance to center [m]')
            ax.set_title('Azimuth residues {:.3}<zen<{:.3}'.format(zen_bin_edges[i], zen_bin_edges[i+1]))

            ax.set_xlabel('phi_gt')
            ax.set_ylabel('phi_pred - phi_gt')
            ax.set_ylim((-3, 3))
            ax.set_ylim((-1, 1))
            fig.tight_layout()
            plt.savefig(os.path.join(self.plot_path, 'az_residues_vs_azimuth_DC2_vs_coredistance_{:.3}_zen_{:.3}.png').format(zen_bin_edges[i], zen_bin_edges[i+1]))
 
    def plot_scatter_uncertainties(self):

        fig, ax = plt.subplots(1, 1)
        sc = ax.scatter(2*self.sig_theta_gt, 3*self.sig_phi_gt, c=self.n_ant, s=3)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(1e-2, 10)
        ax.set_xlim(4e-2, 4)
        fig.colorbar(sc, label='# antennas')
        ax.set_title('PWF uncertainties vs #antennas')
        ax.set_xlabel('sigma_theta')
        ax.set_ylabel('sigma_phi')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'scatter_uncertainties_all_{}.png'.format(self.layout_name)))

        fig, ax = plt.subplots(1, 1)
        sc = ax.scatter(self.theta_gt, 3*self.sig_phi_gt, c=self.n_ant, s=3)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(1e-2, 4)
        fig.colorbar(sc, label='# antennas')
        ax.set_title('azimuth uncertainties vs zenith')
        ax.set_xlabel('zenith gt [deg]')
        ax.set_ylabel('sigma_phi [deg]')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'scatter_phi_uncertainties_vs_zen_{}.png'.format(self.layout_name)))

        fig, ax = plt.subplots(1, 1)
        sc = ax.scatter(self.phi_gt, 3*self.sig_phi_gt, c=self.n_ant, s=3)
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.set_ylim(1e-2, 4)
        fig.colorbar(sc, label='# antennas')
        ax.set_title('azimuth uncertainties vs azimuth')
        ax.set_xlabel('azimuth gt [deg]')
        ax.set_ylabel('sigma_phi [deg]')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'scatter_phi_uncertainties_vs_az_{}.png'.format(self.layout_name)))

        fig, ax = plt.subplots(1, 1)
        sc = ax.scatter(self.theta_gt, 2*self.sig_theta_gt, c=self.n_ant, s=3)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim(4e-2, 4)
        fig.colorbar(sc, label='# antennas')
        ax.set_title('zenith uncertainties vs zenith')
        ax.set_xlabel('zenith gt [deg]')
        ax.set_ylabel('sigma_theta [deg]')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'scatter_theta_uncertainties_vs_zen_{}.png'.format(self.layout_name)))

        fig, ax = plt.subplots(1, 1)
        sc = ax.scatter(self.phi_gt, 2*self.sig_theta_gt, c=self.n_ant, s=3)
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.set_ylim(4e-2, 4)
        fig.colorbar(sc, label='# antennas')
        ax.set_title('zenith uncertainties vs azimuth')
        ax.set_xlabel('azimuth gt [deg]')
        ax.set_ylabel('sigma_theta [deg]')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'scatter_theta_uncertainties_vs_az_{}.png'.format(self.layout_name)))



    def compute_effective_area(self, binning, cst_A_Sim=None, varing_area=False, core_contained=False):

        def area(zen):
            hexagon_size = 12500 * (1/np.cos(np.radians(zen))-1)
            hex_area = (3*np.sqrt(3)*hexagon_size*hexagon_size/2)
            return hex_area

        ert = self.event_res_tab

        #### first if CC only consider the event that fall with the varying area
        if core_contained:
            zenith = ert[:, 2]
            hex_size_max = 8100
            hex_size = [np.min([hex_size_max, 12500 * (1/np.cos(np.radians(zen))-1) ]) for zen in zenith]

            is_in_8100 = [hexy.is_inside_hex([-y, x], hex_s) for (x, y, hex_s) in zip(ert[:, 4], ert[:, 5], hex_size)]

            ert = ert[is_in_8100]
            hex_size = np.array(hex_size)[is_in_8100]

        digitize_zenith = np.digitize(ert[:, 2], binning.zen_bin_edges) - 1
        digitize_energy = np.digitize(ert[:, 1], binning.energy_bin_edges) - 1

        trig_fraction = np.zeros((binning.n_bin_zen, binning.n_bin_energy))
        nb_trigged = np.zeros((binning.n_bin_zen, binning.n_bin_energy))
        weight_map = np.zeros((binning.n_bin_zen, binning.n_bin_energy))

        effective_area = np.zeros((binning.n_bin_zen, binning.n_bin_energy))

        for i_zen in range(binning.n_bin_zen):
            for i_ener in range(binning.n_bin_energy):

                id_inbin = np.where((digitize_zenith == i_zen) *(digitize_energy == i_ener))[0]
                event_number_list = ert[id_inbin, 0]
                weight = ert[id_inbin, 7]
                zenith = ert[id_inbin, 2]

                if varing_area:
                    var_area = area(zenith)
                    weight = weight/var_area

                    if core_contained:
                        hx_size = hex_size[id_inbin]
                        weight_ = ert[id_inbin, 7] * 0
                        var_area_ = var_area * 0

                        for k, ev in enumerate(event_number_list):
                            var_area_[k] = (3*np.sqrt(3)*hx_size[k]*hx_size[k]/2)
                            weight_[k] = 1
                            # check if the tested core lies in the hexagon with hexagon_size = min (8100, hexagon_size(zen))
                            #open the .json file to get the tested core_position

                            json_file = os.path.join(self.files_dir, '{}.json'.format(int(ev)))
                            with open(json_file, 'r') as f:
                                event_params = json.load(f)
                            tested_cores_position = np.array(event_params['tested_cores'])
                            for tcp in tested_cores_position:
                                if hexy.is_inside_hex([-tcp[1], tcp[0]], hx_size[k]):
                                    weight_[k] +=1

                            weight_[k] = weight_[k]/var_area_[k]  
                        weight = weight_

                nb_event_trigged = np.sum(ert[id_inbin, 9] >= self.n_trig_thres)
                nb_trigged[i_zen, i_ener] = np.sum(ert[id_inbin, 9] >= self.n_trig_thres)
                nb_event_total = np.sum(weight)
                trig_fraction[i_zen, i_ener] = nb_event_trigged / nb_event_total
                if nb_event_total == 0:
                    trig_fraction[i_zen, i_ener] = 0

        self.trig_fraction = trig_fraction

        ## need to add the cos(theta) factor the get the ffective area

        for i_zen in range(binning.n_bin_zen):
            effective_area[i_zen, :] = self.trig_fraction[i_zen, :] * np.cos(D2R*binning.zen_bin_centers[i_zen])

        if cst_A_Sim:
            effective_area *= cst_A_Sim
        self.effective_area = effective_area

    def compute_event_rate(self, binning):

        rate_per_day_per_m2 = self.trig_fraction.copy()*0.0

        for i_zen in range(binning.n_bin_zen):
            for i_ener in range(binning.n_bin_energy):
                rate_per_day_per_m2[i_zen, i_ener] = (
                    self.effective_area[i_zen, i_ener] *
                    tale.tale_diff_flux(10**binning.energy_bin_centers[i_ener]) *
                    2*np.pi * np.sin(D2R*binning.zen_bin_centers[i_zen]) *
                    binning.delta_zen[i_zen] * D2R * binning.delta_energy[i_ener] * 60 * 60 * 24
                )

        self.rate_per_day_per_m2 = rate_per_day_per_m2

    def plot_binned_quantities(self, binning):
        if not hasattr(self, 'effective_area'):
            self.compute_effective_area(binning)
        if not hasattr(self, 'rate_per_day_per_m2'):
            self.compute_event_rate(binning)

        plt.figure(figsize=(10, 8))
        pc1 = plt.pcolor(binning.energy_bin_edges, binning.zen_bin_edges,(self.trig_fraction), vmax=1)
        plt.xlabel('log10 E/eV')
        plt.ylabel('Zenith [deg]')
        clb = plt.colorbar(pc1).set_label(label='Trigger fraction')
        plt.title('Trigger fraction for {}'.format(self.layout_name))
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'trigger_fraction_DC2_{}.png'.format(self.layout_name)))

        plt.figure(figsize=(10, 8))
        pc1 = plt.pcolor(binning.energy_bin_edges, binning.zen_bin_edges, (self.rate_per_day_per_m2*self.A_sim))
        plt.xlabel('log10 E/eV')
        plt.ylabel('Zenith [deg]')
        clb = plt.colorbar(pc1).set_label(label='# event/day')
        plt.title('Event rate for {}'.format(self.layout_name))
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'event_rate_DC2_{}.png'.format(self.layout_name)))


        plt.figure(figsize=(10, 8))
        pc1 = plt.pcolor(binning.energy_bin_edges, binning.zen_bin_edges,np.log10(self.trig_fraction), vmax=1)
        plt.xlabel('log10 E/eV')
        plt.ylabel('Zenith [deg]')
        clb = plt.colorbar(pc1).set_label(label='log10 Trigger fraction')
        plt.title('Trigger fraction for {}'.format(self.layout_name))
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'trigger_fraction_DC2_{}_log10.png'.format(self.layout_name)))

        plt.figure(figsize=(10, 8))
        pc1 = plt.pcolor(binning.energy_bin_edges, binning.zen_bin_edges, np.log10(self.rate_per_day_per_m2*self.A_sim))
        plt.xlabel('log10 E/eV')
        plt.ylabel('Zenith [deg]')
        clb = plt.colorbar(pc1).set_label(label='log10 # event/day')
        plt.title('Event rate for {}'.format(self.layout_name))
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'event_rate_DC2_{}_log10.png'.format(self.layout_name)))

        plt.figure(figsize=(10, 8))
        pc2 = plt.pcolor(binning.energy_bin_edges, binning.zen_bin_edges,(self.effective_area)/1e6)
        plt.xlabel('log10 E/eV')
        plt.ylabel('Zenith [deg]')
        clb = plt.colorbar(pc2).set_label(label='Effective area [km2]')
        plt.title('Effective area for {}'.format(self.layout_name))
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'effective_area_DC2_{}.png'.format(self.layout_name)))


