import grand.dataio.root_trees as groot
#import grand.grandlib_classes.grandlib_classes as glc
# import the rest of the guardians of the galaxy:
import numpy as np
import matplotlib.pyplot as plt
import uproot
import os


class dd(groot.DataDirectory):
    def __init__(self, dir_name: str, recursive: bool = False, analysis_level: int = -1, sim2root_structure: bool = True):
        """
        @param dir_name: the name of the directory to be scanned
        @param recursive: if to scan the directory recursively
        @param analysis_level: which analysis level files to read. -1 means max
        """
        super().__init__(dir_name, recursive, analysis_level, sim2root_structure)

        self.event_list = self.tefield_l0.get_list_of_events()

        self.nb_event = len(self.event_list)
        self.current_event_number = 0
        self.is_run_set = False
        self.load_event_stats()

    def load_event_stats(self):

        self.up_tshower_l0 = uproot.open(self.ftshower_l0.filename)["tshower"]
        self.up_tshowersim = uproot.open(self.ftshowersim.filename)["tshowersim"]
        self.all_energies = self.up_tshower_l0['energy_primary'].array().to_numpy()
        self.all_zenith = self.up_tshower_l0['zenith'].array().to_numpy()
        self.all_azimuth = self.up_tshower_l0['azimuth'].array().to_numpy()
        self.all_xmax_grams = self.up_tshower_l0['xmax_grams'].array().to_numpy()
        self.all_event_number = self.up_tshower_l0['event_number'].array().to_numpy()
        self.all_primary_types = self.up_tshower_l0['primary_type'].array().to_numpy()
        self.all_event_weights = self.up_tshowersim['event_weight'].array().to_numpy()
        self.all_xmax_pos_shc = self.up_tshower_l0['xmax_pos_shc'].array().to_numpy()

    def plot_stats_plots(self, savedir):

        p_id = np.where(self.all_primary_types == '2212')[0]
        fe_id = np.where(self.all_primary_types == 'Fe^56')[0]

        plt.figure(1)
        plt.clf()
        plt.hist(self.all_xmax_grams[p_id], bins=20, range=[400, 1200], alpha=0.5, label='protons')
        plt.hist(self.all_xmax_grams[fe_id], bins=20, range=[400, 1200], alpha=0.5, label='Fe56')
        plt.legend()
        plt.title('Xmax distribution for p and Fe')
        plt.xlabel('Xmax [g.cm^-2]')
        plt.ylabel('# events')
        plt.savefig(os.path.join(savedir, 'xmax_p_fe.png'))

        plt.figure(2)
        plt.clf()
        plt.hist(np.log10(self.all_energies[p_id]), range=[7, 10], bins=20, alpha=0.5, label='protons')
        plt.hist(np.log10(self.all_energies[fe_id]), range=[7, 10], bins=20, alpha=0.5, label='Fe56')
        plt.legend()
        plt.title('Energy distribution for p and Fe')
        plt.xlabel('Energy primary [GeV]')
        plt.ylabel('# events')
        plt.savefig(os.path.join(savedir, 'Energy_p_fe.png'))


    def get_event(self, event_number, run_number):
        if event_number != self.current_event_number:
            self.tefield_l0.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
            self.tshower_l0.get_event(event_number, run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
            self.tefield_l1.get_event(event_number, run_number)           # update traces, du_pos etc for event with event_idx.
            self.tvoltage_l0.get_event(event_number, run_number)
            self.tadc_l1.get_event(event_number, run_number)
            self.current_event_number = event_number
        if not self.is_run_set:
            self.trun_l0.get_run(run_number)                         # update run info to get site latitude and longitude.
            self.trun_l1.get_run(run_number)                         # update run info to get site latitude and longitude.            
            self.trunefieldsim_l0.get_run(run_number)
            self.trunefieldsim_l1.get_run(run_number)       
            self.is_run_set = True

            self.t_pre_l0 = self.trunefieldsim_l0.t_pre
            self.t_pre_l1 = self.trunefieldsim_l1.t_pre

        ev_energy = self.tshower_l0.energy_primary
        ev_zenith = self.tshower_l0.zenith
        ev_azimuth = self.tshower_l0.azimuth
        ev_xmax_grams = self.tshower_l0.xmax_grams
        ev_primary_type = self.tshower_l0.primary_type

        self.event_params = {}
        self.event_params['energy'] = str(ev_energy)
        self.event_params['azimuth'] = str(ev_azimuth)
        self.event_params['zenith'] = str(ev_zenith)
        self.event_params['xmax_grams'] = str(ev_xmax_grams)
        self.event_params['primary_type'] = str(ev_primary_type)
        self.event_params['event_number'] = str(event_number)
        self.event_params['event_second'] = (self.tshower_l0.core_time_s)
        self.event_params['event_nsecond'] = (self.tshower_l0.core_time_ns)
        self.event_params['shower_core_x'] = str(self.tshower_l0.shower_core_pos[0])
        self.event_params['shower_core_y'] = str(self.tshower_l0.shower_core_pos[1])
        self.event_params['shower_core_z'] = str(self.tshower_l0.shower_core_pos[2])
        # event_weight is not loaded by the DataDirectory code so neet to get it from the uproot
        id_up = np.where(self.all_event_number == self.current_event_number)[0][0]
        self.event_params['event_weight'] = str(self.all_event_weights[id_up])
        self.event_params['xmax_pos_x'] = str(self.tshower_l0.xmax_pos_shc[0])
        self.event_params['xmax_pos_y'] = str(self.tshower_l0.xmax_pos_shc[1])
        self.event_params['xmax_pos_z'] = str(self.tshower_l0.xmax_pos_shc[2])

    def load_event(self, event_number, run_number, ef_l1_mode=True):
        if event_number != self.current_event_number:
            self.get_event(event_number, run_number)

        # this gives the indices of the antennas of the array participating in this event
        self.ev_dus_indices = self.tefield_l0.get_dus_indices_in_run(self.trun_l0)

        self.ev_du_list = np.asarray(self.trun_l0.du_id)[self.ev_dus_indices]
        self.ev_du_position = np.asarray(self.trun_l0.du_xyz)[self.ev_dus_indices]

        # t0 calculations
        self.ev_second = self.event_params['event_second']
        self.ev_nsecond = self.event_params['event_nsecond']

        self.ev_t0_efield_l1 = (self.tefield_l1.du_seconds-self.ev_second)*1e9  - self.ev_nsecond + self.tefield_l1.du_nanoseconds 
        if not ef_l1_mode:
            self.ev_t0_efield_l0 = (self.tefield_l0.du_seconds-self.ev_second)*1e9  - self.ev_nsecond + self.tefield_l0.du_nanoseconds
            self.ev_t0_adc_l1 = (self.tadc_l1.du_seconds-self.ev_second)*1e9  - self.ev_nsecond + self.tadc_l1.du_nanoseconds
            self.ev_t0_voltage_l0 = (self.tvoltage_l0.du_seconds-self.ev_second)*1e9  - self.ev_nsecond + self.tvoltage_l0.du_nanoseconds


        # loading the traces as numpy arrays
        if not ef_l1_mode:
            self.ev_trace_efield_l0 = np.asarray(self.tefield_l0.trace, dtype=np.float32)   # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)
            self.ev_trace_voltage = np.asarray(self.tvoltage_l0.trace, dtype=np.float32)
            self.ev_trace_ADC_l1 = np.asarray(self.tadc_l1.trace_ch, dtype=np.float32)
        self.ev_trace_efield_l1 = np.asarray(self.tefield_l1.trace, dtype=np.float32)
        
        #du_id = np.asarray(tefield_l0.du_id) # MT: used for printing info and saving in voltage tree.

        self.ev_dt_ns_l0 = np.asarray(self.trun_l0.t_bin_size)[self.ev_dus_indices]  # sampling time in ns, sampling freq = 1e9/dt_ns. 
        self.ev_dt_ns_l1 = np.asarray(self.trun_l1.t_bin_size)[self.ev_dus_indices]  # sampling time in ns, sampling freq = 1e9/dt_ns. 

        if not ef_l1_mode:
            self.ev_trace_efield_l0_time = np.arange(0, self.ev_trace_efield_l0.shape[-1]) * self.ev_dt_ns_l0[0] - self.t_pre_l0
            self.ev_trace_voltage_time = np.arange(0, self.ev_trace_voltage.shape[-1]) * self.ev_dt_ns_l0[0] - self.t_pre_l0
            self.ev_trace_ADC_l1_time = np.arange(0, self.ev_trace_ADC_l1.shape[-1]) * self.ev_dt_ns_l1[0] - self.t_pre_l1
        self.ev_trace_efield_l1_time = np.arange(0, self.ev_trace_efield_l1.shape[-1]) * self.ev_dt_ns_l1[0] - self.t_pre_l1
