from import_utils import loadmat_h5, _load
import import_utils
import toolbox as tb
import numpy as np
import os

# best probes for Neuropixel data
DEFAULT_PROBES = [1, 3, 4]
# time stamps for stimulation and blanking
STIM_WINDOW = (5, 105)
# blk window uses the last 50 ms of resp_train_blk
BLK_WINDOW = (106, 155)

BLKDUR = BLK_WINDOW[1] - BLK_WINDOW[0]
STIMDUR = STIM_WINDOW[1] - STIM_WINDOW[0]


class Session:
    """
    Initiate from session .mat file
    """

    def __init__(self, mat, type="utah", neuron_exclusion=True):
        """
        Parameters:
        -----------
        mat : str
            path to session.mat file
        probe : int
            if not None, use the given probe
        type : str
            "utah" : if session has Utah array recordings
            "neuropixel" : if session has neuropixel recordings
        neuron_exclusion : bool
            Default True. Excludes neurons based on criteria defined in self.neuron_exclusion()
        """
        temp = import_utils.loadmat_h5(mat)
        self.__dict__ = temp["Session"]
        self.d = temp["Session"]
        self.fields = list(self.d.keys())
        # probes are 1-indexed NOT 0-indexed so we must subtract 1
        if type == "utah":
            self.use_probe(num=None)
            if neuron_exclusion:
                self.neuron_exclusion()
        elif type == "neuropixel":
            print("use use_probe function to select one or multiple probes")
            self.use_probe(num=DEFAULT_PROBES)

    def use_probe(self, num=1) -> None:
        if num is not None:
            if type(num) is list:
                probe_nums = [val - 1 for val in num]
                trials = []
                to_concat_resp_large = []
                to_concat_resp_small = []
                to_concat_xy = []

                for probe in probe_nums:
                    trials.append(self.d["T"][0][probe][0][0])
                    to_concat_resp_large.append(self.d["resp_large"][0][probe])
                    to_concat_resp_small.append(self.d["resp_small"][0][probe])
                    to_concat_xy.append(self.d["XYch"][0][probe])
                self.n_trials = trials
                self.resp_large = np.concatenate(to_concat_resp_large, axis=0)
                self.resp_small = np.concatenate(to_concat_resp_small, axis=0)
                self.xy_coords = np.concatenate(to_concat_xy, axis=0)
                self.np_coords = self._get_neuron_np_coords()
            else:
                self.probe = num - 1
                self.n_trials = self.d["T"][0][self.probe][0][0]
                # data from all probes below
                self.resp_large = self.d["resp_large"][0][self.probe]
                self.resp_small = self.d["resp_small"][0][self.probe]
                self.xy_coords = self.d["XYch"][0][self.probe]
                self.np_coords = self._get_neuron_np_coords()
        else:  # for utah array data
            self.n_trials = self.d["T"][0][0]
            self.resp_large = self.d["resp_large"]
            self.resp_small = self.d["resp_small"]
            self.xy_coords = self.d["XYch"]
            self.n_neurons = len(self.xy_coords)
            self.np_coords = self._get_neuron_np_coords()

    def get_center_neurons(self, thresh=25, d=10):
        """
        Splits neurons into three pools: center, off-center, and excised.

        Parameters:
        -----------
        thresh : int
            Neurons within this radius (in pixels) are considered to be "centered"
        d : int
            Neurons between this radius and thresh (in pixels) are considered on the border and are excised.
            Neurons outside of the radius thresh + d are considered off-center 
        
        Returns:
        -----------
        self.idx_d : dict
            contains the indexes of the neurons that are 'center','off_center',and 'excised'
        """
        out = tb.is_centered_xy(self.XYch, thresh=thresh, d=d)

        centered_neurons = np.where(np.asarray(out) == 1)[0]
        off_center_neurons = np.where(np.asarray(out) == 2)[0]
        excised_neurons = np.where(np.asarray(out) == 0)[0]

        self.idx_d = {'center':centered_neurons,'off_center':off_center_neurons,'excised':excised_neurons}

    def _get_neuron_np_coords(self, transform=True):
        """
        Extract (x,y) coordinate from Session data

        Parameters:
        -----------
        transform : bool
            if True, transforms neuron coordinates (where origin is center of image) to numpy readable coordinates (where origin is top left)

        Raises:
        ------
        self.coords : dict
            if self.probe is None:
                {'probe1': list of coords, 'probe2':...} for however many probes there are
        self.coords : list
            if self.probe:
                list of coords for the given probe
        """
        _transform = lambda x: tb.transform_coord_system(x) if transform else x
        coords_list = self.xy_coords
        coords = [_transform(item) for item in coords_list]

        return coords
    
    def _get_response(self):
        pass
        
    def _get_thresholds(self,alpha=1):
        """
        Returns the thresholds for response depending on alpha 

        Parameters: 
        ---------------

        alpha : float 
            a multiplier for how many standard deviations above spontaneous activity is considered responsive
        """
        n_neurons,n_images,n_trials,n_ms = self.resp_train[:,0,:,:].shape
        blkdur = BLKDUR

        trains_blk = self.resp_train_blk[:,:,:,n_ms-blkdur:n_ms+1]
        spike_count_blk = np.sum(trains_blk,axis=-1)
        rate_blk = spike_count_blk/blkdur

        rate_blk_mean = np.mean(rate_blk,axis=(-1,-2))
        rate_blk_std = np.std(rate_blk,axis=(-1,-2))

        thresh = rate_blk_mean + alpha*rate_blk_std

        return thresh


    def get_image_data(self, img_idx):
        """
        Extracts large and small image neural responses from a particular image

        Parameters:
        ------------
        img_idx : int
            Index of which image to extract data from
        """
        resp_large = self.resp_large[:, img_idx, :]
        resp_small = self.resp_small[:, img_idx, :]
        neuron_coords = self._get_neuron_np_coords()
        d = {
            "resp_large": resp_large,
            "resp_small": resp_small,
            "coords": neuron_coords,
        }
        return d

    def neuron_exclusion(self, rate_std_thresh=1, large_img_idx=1):
        """
        Excludes neurons for a given image, exclusion criteria are:

        i) rate_evoked n std. devs above rate_spontaneous
        ii) fano factor < 5

        Parameters:
        ------------
        rate_std_thresh : float
            a multiplier that determines threshold strictness for driven activity
        """

        n_neurons, n_images, n_trials, n_ms = self.resp_train[:, 0, :, :].shape

        blkdur = BLK_WINDOW[1] - BLK_WINDOW[0]

        # splice together train
        # neurons x images x trials x ms
        a_large = self.resp_train[:, large_img_idx, :, :, :]
        a_small = self.resp_train[:, large_img_idx - 1, :, :, :]
        # neurons x images x trials x last 50 ms of blank
        b = self.resp_train_blk[:, :, :, n_ms - blkdur : n_ms + 1]

        spike_count_stim_large = np.sum(a_large, axis=-1)
        spike_count_stim_small = np.sum(a_small, axis=-1)
        spike_count_blk = np.sum(b, axis=-1)

        # rate and spontaneous activity calculation

        rate_stim_large = spike_count_stim_large / (STIM_WINDOW[1] - STIM_WINDOW[0])
        rate_stim_small = spike_count_stim_small / (STIM_WINDOW[1] - STIM_WINDOW[0])
        rate_blk = spike_count_blk / (BLK_WINDOW[1] - BLK_WINDOW[0])

        rate_blk_mean = np.mean(rate_blk, axis=(-1, -2))
        rate_blk_std = np.std(rate_blk, axis=(-1, -2))

        rate_thresh = rate_blk_mean + rate_std_thresh * (rate_blk_std)
        rate_thresh_full = np.expand_dims(rate_thresh, -1).repeat(n_images, axis=-1)

        # LARGE IMAGE EXCLUSION
        rate_stim_mean_large = np.mean(rate_stim_large, axis=-1)
        # excludes (neuron,image) pairs with low evoked activity
        cond_rate_large = rate_stim_mean_large < rate_thresh_full

        mm_large = np.mean(spike_count_stim_large, axis=-1)
        vv_large = (np.std(spike_count_stim_large, axis=-1)) ** 2

        ff_large = mm_large / vv_large
        # SMALL IMAGE EXCLUSION
        rate_stim_mean_small = np.mean(rate_stim_small, axis=-1)
        # excludes (neuron,image) pairs with high evoked activity
        cond_rate_small = rate_stim_mean_small > rate_thresh_full

        mm_small = np.mean(spike_count_stim_small, axis=-1)
        vv_small = (np.std(spike_count_stim_small, axis=-1)) ** 2

        ff_small = mm_small / vv_small

        cond_ff_large = ff_large > 5
        cond_ff_small = ff_small > 5

        excluded_neuron_idx_large = np.concatenate(
            (np.where(cond_rate_large)[0], np.where(cond_ff_large)[0])
        )
        excluded_image_idx_large = np.concatenate(
            (np.where(cond_rate_large)[1], np.where(cond_ff_large)[1])
        )

        excluded_neuron_idx_small = np.concatenate(
            (
                np.where(cond_rate_small)[0],
                np.where(cond_rate_large)[0],
                np.where(cond_ff_small)[0],
                np.where(cond_ff_large)[0],
            )
        )
        excluded_image_idx_small = np.concatenate(
            (
                np.where(cond_rate_small)[1],
                np.where(cond_rate_large)[1],
                np.where(cond_ff_small)[1],
                np.where(cond_ff_large)[1],
            )
        )

        excluded_image_idx_large = np.unique(excluded_image_idx_large)
        excluded_image_idx_small = np.unique(excluded_image_idx_small)

        self.resp_large[excluded_neuron_idx_large, excluded_image_idx_large, :] = np.nan
        self.resp_small[excluded_neuron_idx_small, excluded_image_idx_large, :] = np.nan

        exclusion_map_small = np.ones((n_neurons, n_images))
        exclusion_map_large = np.ones((n_neurons, n_images))
        exclusion_map_small[excluded_neuron_idx_small, excluded_image_idx_small] = 0
        exclusion_map_large[excluded_neuron_idx_large, excluded_image_idx_large] = 0
        self.exclusion_map_small = exclusion_map_small
        self.exclusion_map_large = exclusion_map_large

    # Decide whether to put spike-train utils here as staticmethods or in another module
    @staticmethod
    def _get_spike_counts(train):
        a = train
        out = a / a[np.nonzero(a)].min()

        return out

    @staticmethod
    def _get_firing_rate(counts, window):
        t1, t2 = window
        r = sum(counts[t1 : t2 + 1]) / (t2 - t1)

        return r
