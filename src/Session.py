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

    def get_neuron_locations(self, thresh=25, d=10):
        """
        Splits neurons into three pools: center, off-center, and excised.

        Parameters:
        -----------
        thresh : int
            Neurons within this radius (in pixels) are considered to be "centered"
        d : int
            Neurons between this radius and thresh (in pixels) are considered on the border and are excised.
            Neurons outside of the radius thresh + d are considered off-center

        Raises:
        -----------
        self.neuron_locations : dict
            contains the indexes of the neurons that are 'center','off_center',and 'excised'
        self.resp_train_center : ndarray
            the response spike trains of centered neuron
        """
        out = tb.is_centered_xy(self.XYch, thresh=thresh, d=d)
        locations = ["excised", "center", "off_center"]
        location_codes = [0, 1, 2]

        temp = zip(locations, location_codes)

        self.neuron_locations = {k: np.where(np.asarray(out) == v)[0] for k, v in temp}

        resp_train_reindex = np.moveaxis(self.resp_train, 1, 2)

        self.resp_train_d = {
            k: resp_train_reindex[self.neuron_locations[k], ...] for k in locations
        }

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

    def neuron_exclusion(self, thresh=25, d=10, alpha=1, unresponsive_alpha=0):
        
        self._neuron_exclusion(
            thresh=thresh,
            d=d,
            alpha=alpha,
            unresponsive_alpha=unresponsive_alpha
            )

        _add_dims = lambda x,y: np.reshape(x,list(x.shape)+[1]*(y.ndim-x.ndim))

        locations = ["center", "off_center"]

        x = self.exclusion_masks
        y = self.resp_train_d

        extended_exclusion_maps = {k:_add_dims(x[k],y[k]) for k in locations
        }

        exclusion_maps_br = {
            k:np.broadcast_to(extended_exclusion_maps[k],y[k].shape) for k in locations
            }
        
        z = exclusion_maps_br
        
        self.resp_train_masked = {
            k:np.ma.masked_array(y[k],mask=z[k]) for k in locations
        }

    def _neuron_exclusion(self, thresh=25, d=10, alpha=1, unresponsive_alpha=0):
        self.get_neuron_locations(thresh=thresh,d=d)
        self.get_mean_fr()
        self.get_thresholds(alpha=alpha, unresponsive_alpha=unresponsive_alpha)

        temp = {
            "center":(self.mean_frs["center_small"] < self.thresholds["center_responsive"]),
            "off_center_1":self.mean_frs["off_center_small"] > self.thresholds["off_center_unresponsive"],
            "off_center_2":self.mean_frs["off_center_large"] < self.thresholds["off_center_responsive"]
        }

        self.exclusion_masks = {
            "center" : temp["center"],
            "off_center" : np.logical_and(temp["off_center_1"],temp["off_center_2"])
        }

    def get_mean_fr(self):
        locations = ["center", "off_center", "off_center"]
        im_sizes = ["small", "large", "small"]

        self.mean_frs = {
            k + "_" + v: self._get_mean_fr(neuron_location=k, im_size=v)
            for k, v in zip(locations, im_sizes)
        }

    def _get_mean_fr(self, neuron_location="center", im_size=None):
        """
        Calculates the mean firing rate of neurons in the location specified by neuron_location.
        Firing rates are calculated as a response to either the large or small image.

        Parameters:
        -----------
        neuron_location : str
            "center" : get firing rates for only neurons in the center.
            "off_center" : get firing rates for neurons off the center
        im_size : str or None
            None : default behavior, uses small stimulus for center neurons and large stimulus for off-center neurons
            "small" : uses neuron responses to the small stimulus
            "large" : uses neuron responses to the large stimulus
        """

        d = {"small": 0, "large": 1}

        if neuron_location == "center":  # use small image
            img_idx = 0
        elif neuron_location == "off_center":  # use large image
            img_idx = 1

        if im_size is not None:
            img_idx = d[im_size]
        else:
            pass

        _train_stim = self.resp_train[
            self.neuron_locations[neuron_location], img_idx, :, :, :
        ]
        _spike_count_stim = np.sum(_train_stim, axis=-1)
        _rate_stim = _spike_count_stim / STIMDUR
        _rate_stim_mean = np.mean(_rate_stim, axis=-1)

        out = _rate_stim_mean

        return out

    def get_thresholds(self, alpha=1, unresponsive_alpha = 0):
        locations = ["center", "off_center","off_center"]
        conditions = ["_responsive","_responsive","_unresponsive"]
        alphas = [alpha, alpha, unresponsive_alpha]
        self.thresholds = {
            i+j: self._get_threshold(alpha=k, neuron_location=i)
            for i,j,k in zip(locations, conditions, alphas)
        }

    def _get_threshold(self, alpha=1, neuron_location="center"):
        """
        Returns the thresholds for response depending on alpha

        Parameters:
        ---------------

        alpha : float
            a multiplier for how many standard deviations above spontaneous activity is considered responsive
        """
        n_neurons, n_images, n_trials, n_ms = self.resp_train[:, 0, :, :].shape
        blkdur = BLKDUR

        trains_blk = self.resp_train_blk[:, :, :, n_ms - blkdur : n_ms + 1]
        spike_count_blk = np.sum(trains_blk, axis=-1)
        rate_blk = spike_count_blk / blkdur

        rate_blk_mean = np.mean(rate_blk, axis=(-1, -2))
        rate_blk_std = np.std(rate_blk, axis=(-1, -2))

        thresh = rate_blk_mean + alpha * rate_blk_std

        thresh = thresh[self.neuron_locations[neuron_location]]

        thresh_full = thresh[..., np.newaxis].repeat(n_images, axis=-1)

        out = thresh_full

        return out

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
