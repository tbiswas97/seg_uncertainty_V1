from import_utils import loadmat_h5, _load
import import_utils
import toolbox as tb
import numpy as np
import os
from itertools import combinations
import pandas as pd
from collections import Counter

# best probes for Neuropixel data
DEFAULT_PROBES = [1, 3, 4]
# time stamps for stimulation and blanking
STIM_WINDOW = (0, 105)
# blk window uses the last 50 ms of resp_train_blk
BLK_WINDOW = (105, 155)
SMALL_LARGE_IDXS = {"small": 0, "large": 1}

BLKDUR = BLK_WINDOW[1] - BLK_WINDOW[0]
STIMDUR = STIM_WINDOW[1] - STIM_WINDOW[0]


class Session:
    """
    Initiate from session .mat file
    """

    def __init__(self, mat, _type="utah", neuron_exclusion=True):
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
        if mat.split(".")[-1] == "pkl":
            temp = import_utils._load(mat)
        else:
            temp = import_utils.loadmat_h5(mat)
        self.__dict__ = temp["Session"]
        self.d = temp["Session"]
        self.fields = list(self.d.keys())
        self._type = _type
        # probes are 1-indexed NOT 0-indexed so we must subtract 1
        if _type == "utah":
            self.use_probe(num=None)
            self.get_exp_info()
            if neuron_exclusion:
                self.neuron_exclusion()
        elif _type == "neuropixel":
            # print("use use_probe function to select one or multiple probes")
            data_keys = []
            for key in self.__dict__.keys():
                if type(self.__dict__[key]) == np.ndarray:
                    data_keys.append(key)
            data_keys.remove("NOTES")
            for key in data_keys:
                self.__dict__[key] = self.__dict__[key].squeeze()
                self.__dict__[key] = np.concatenate([slc for slc in self.__dict__[key]])
            self.xy_coords = self.XYch
            self.np_coords = self._get_neuron_np_coords()

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

    def get_exp_info(self):
        n_neurons, n_conditions, n_images, n_trials, n_ms = self.resp_train[
            :, :, :, :, :
        ].shape
        self.exp_info = {
            "n_neurons": n_neurons,
            "n_conditions": n_conditions,
            "n_images": n_images,
            "n_trials": n_trials,
            "n_ms": n_ms,
        }

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
        out = tb.is_centered_xy(self.XYch, origin=(0, 0), thresh=thresh, d=d)
        locations = ["excised", "center", "off_center"]
        location_codes = [0, 1, 2]

        temp = zip(locations, location_codes)

        self.neuron_locations = {k: np.where(np.asarray(out) == v)[0] for k, v in temp}

        self.spike_counts = np.sum(self.resp_train, axis=-1)

        self.resp_train_d = {
            k: self.resp_train[self.neuron_locations[k], ...] for k in locations
        }

    def _get_neuron_location(self, neuron):
        for k in self.neuron_locations.keys():
            if neuron in self.neuron_locations[k]:
                return k

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

    def neuron_exclusion(
        self, thresh=25, d=10, alpha=1, unresponsive_alpha=0, mr_thresh=0.9
    ):
        """
        Main function that excludes neurons based on the described criteria:
        The responsiveness threshold is:

        threshold = Rsc_spontaneous_mean + alpha*Rsc_spontaenous_std

        thresh : float
            a distance threshold below which neurons are considered to be "center"
        d : float
            (thresh+d) is the distance threshold above which neurons are considered to be "off_center"
        alpha : float
            excluded neurons are those with activity below
        unresponsive_alpha : float
            included neurons that should be unresponsive
        mr_thresh : float
            a threshold for neuron modulation ratios

        """
        # neuron exclusions
        self.neuron_exclusion_parameters = {
            "thresh": thresh,
            "d": d,
            "alpha": alpha,
            "unresponsive_alpha": unresponsive_alpha,
            "mr_thresh": mr_thresh,
        }
        self.get_neuron_locations(thresh=thresh, d=d)

        self.get_mean_sc()
        self.get_mean_fr()
        self.get_thresholds(alpha=alpha, unresponsive_alpha=unresponsive_alpha)

        if mr_thresh is not None:
            self.neuron_locations_mr = {}
            self.get_neuron_modulation_ratio()
            self.neuron_locations_mr["center"] = self.neuron_locations["center"][
                (self.MM_large / self.MM_small)[
                    np.asarray(range(self.exp_info["n_neurons"])),
                    np.argmax(self.MM_small, axis=1),
                ][self.neuron_locations["center"]]
                <= 0.75
            ]

            self.neuron_locations_mr["off_center"] = self.neuron_locations[
                "off_center"
            ][
                (self.MM_large / self.MM_small)[
                    np.asarray(range(self.exp_info["n_neurons"])),
                    np.argmax(self.MM_small, axis=1),
                ][self.neuron_locations["off_center"]]
                > 1
            ]
        else:
            self.neuron_locations_mr = None

        # change all mean firing rates and thresholds to spike counts

        # to_compare = np.maximum(
        # self.mean_scs["center_small"], self.mean_scs["center_large"]
        # )
        to_compare = self.mean_frs["center_small"]

        temp_inclusion = {
            # change to center_large OR max(center_large, center_small)
            "center": (to_compare >= self.thresholds["center_responsive"]),
            "off_center_1": self.mean_frs["off_center_small"]
            <= self.thresholds["off_center_unresponsive"],
            "off_center_2": self.mean_frs["off_center_large"]
            >= self.thresholds["off_center_responsive"],
        }

        self.exclusion_masks = {
            "center": ~(temp_inclusion["center"]),
            "off_center": ~(
                np.logical_and(
                    temp_inclusion["off_center_1"], temp_inclusion["off_center_2"]
                )
            ),
        }

        self.masked = True

    def get_df(self):
        to_concat = []
        for im in range(self.exp_info["n_images"]):
            df = self._get_im_df(im)
            to_concat.append(df)

        out = pd.concat(to_concat, ignore_index=True)

        self.df = out

        return out

    def _get_im_df(self, im):
        responsive_neurons = self._get_responsive_neurons_at_image(im)
        responsive_neurons = [int(neuron) for neuron in responsive_neurons]
        pairs = list(combinations(responsive_neurons, 2))

        d = {}
        origin = [0, 0]

        d["img_idx"] = [im] * len(pairs)
        d["pairs"] = pairs
        d["neuron1"] = [pair[0] for pair in pairs]
        d["neuron1_distance_from_origin"] = [
            tb.euclidean_distance(self.XYch[pair[0]], origin) for pair in pairs
        ]
        d["neuron1_pos"] = [self._get_neuron_location(pair[0]) for pair in pairs]
        d["neuron1_xy_coord"] = [self.XYch[pair[0]] for pair in pairs]
        d["neuron1_np_coord"] = [self.np_coords[pair[0]] for pair in pairs]
        d["neuron2"] = [pair[1] for pair in pairs]
        d["neuron2_distance_from_origin"] = [
            tb.euclidean_distance(self.XYch[pair[1]], origin) for pair in pairs
        ]
        d["neuron2_pos"] = [self._get_neuron_location(pair[1]) for pair in pairs]
        d["neuron2_xy_coord"] = [self.XYch[pair[1]] for pair in pairs]
        d["neuron2_np_coord"] = [self.np_coords[pair[1]] for pair in pairs]
        d["rsc_small"] = [self._get_rsc_at_image(pair, im, "small") for pair in pairs]
        d["rsc_large"] = [self._get_rsc_at_image(pair, im, "large") for pair in pairs]
        d["distance"] = [self._get_dist_between_neurons(pair) for pair in pairs]

        df = pd.DataFrame.from_dict(d)

        df["pair_orientation"] = 0
        df.loc[
            (df.neuron1_pos == "center") & (df.neuron2_pos == "center"),
            "pair_orientation",
        ] = 1
        df.loc[
            (df.neuron1_pos == "center") ^ (df.neuron2_pos == "center"),
            "pair_orientation",
        ] = 2
        df.loc[df.pair_orientation == 1, "pair_orientation"] = "centered"
        df.loc[df.pair_orientation == 2, "pair_orientation"] = "mixed"

        df = df.loc[df.pair_orientation != 0]

        counts = Counter(df.img_idx)
        n_pairs = [counts[num] for num in df.img_idx]
        df.insert(1, "n_pairs", n_pairs)

        return df

    def get_neuron_info(self):
        df = self.df
        neuron1 = df.loc[:, ["neuron1", "neuron1_xy_coord", "neuron1_pos"]].rename(
            {"neuron1": "neuron", "neuron1_xy_coord": "xy_coord", "neuron1_pos": "pos"},
            axis=1,
        )
        neuron2 = df.loc[:, ["neuron2", "neuron2_xy_coord", "neuron2_pos"]].rename(
            {"neuron2": "neuron", "neuron2_xy_coord": "xy_coord", "neuron2_pos": "pos"},
            axis=1,
        )

        neurons=pd.concat([neuron1,neuron2],ignore_index=True,axis=0)
        out = neurons.drop_duplicates(["neuron"]).reset_index()

        return out

    def _get_dist_between_neurons(self, neuron_pair_tup):
        neuron1 = neuron_pair_tup[0]
        neuron2 = neuron_pair_tup[1]
        coord1 = self.XYch[neuron1]
        coord2 = self.XYch[neuron2]

        return tb.euclidean_distance(coord1, coord2)

    def _get_rsc_at_image(self, neuron_pair_tup, image, condition):
        """ """
        x, y = self.spike_counts[neuron_pair_tup, SMALL_LARGE_IDXS[condition], image, :]

        out = tb.pearson_r(x, y)

        return out

    def get_neuron_modulation_ratio(self):
        self.modulation_ratios = (self.MM_large / self.MM_small)[
            np.asarray(range(len(self.MM_large))), np.argmax(self.MM_small, axis=1)
        ]

    def _get_neuron_modulation_ratio(self, idx, location):
        if location == "center":
            driving_im = np.argmax(self.mean_frs["center_large"][idx])
            fr_large = self.mean_frs["center_large"][idx][driving_im]
            fr_small = self.mean_frs["center_small"][idx][driving_im]
        elif location == "off_center":
            driving_im = np.argmax(self.mean_frs["off_center_large"][idx])
            fr_large = self.mean_frs["off_center_large"][idx][driving_im]
            fr_small = self.mean_frs["off_center_small"][idx][driving_im]

        mr = fr_large / fr_small

        return mr

    def _get_responsive_neurons_at_image(self, image):
        locations = ["center", "off_center"]

        if self.neuron_locations_mr is not None:
            responsive_idxs = {
                k: np.asarray(
                    list(
                        set(self.neuron_locations_mr[k]).intersection(
                            set(
                                self.neuron_locations[k][
                                    ~(self.exclusion_masks[k][:, image])
                                ]
                            )
                        )
                    )
                )
                for k in locations
            }

        else:
            responsive_idxs = {
                k: self.neuron_locations[k][~(self.exclusion_masks[k][:, image])]
                for k in locations
            }

        out = np.concatenate(list(responsive_idxs.values()))

        return out

    def get_mean_sc(self, use_session_vars=True):
        fr_small = self.MM_small / 1
        fr_large = self.MM_large / 1

        if use_session_vars:
            self.mean_scs = {
                "center_small": fr_small[self.neuron_locations["center"]],
                "center_large": fr_large[self.neuron_locations["center"]],
                "off_center_small": fr_small[self.neuron_locations["off_center"]],
                "off_center_large": fr_large[self.neuron_locations["off_center"]],
            }

    def get_mean_fr(self, use_session_vars=True):
        locations = ["center", "center", "off_center", "off_center"]
        im_sizes = ["small", "large", "small", "large"]

        if use_session_vars:
            fr_small = self.MM_small / STIMDUR
            fr_large = self.MM_large / STIMDUR

            self.mean_frs = {
                "center_small": fr_small[self.neuron_locations["center"]],
                "center_large": fr_large[self.neuron_locations["center"]],
                "off_center_small": fr_small[self.neuron_locations["off_center"]],
                "off_center_large": fr_large[self.neuron_locations["off_center"]],
            }
        else:
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

        d = SMALL_LARGE_IDXS

        if neuron_location == "center":  # use small image
            idx = 0
        elif neuron_location == "off_center":  # use large image
            idx = 1

        if im_size is not None:
            idx = d[im_size]
        else:
            pass

        _train_stim = self.resp_train[
            self.neuron_locations[neuron_location], idx, :, :, :
        ]
        _spike_count_stim = np.sum(_train_stim, axis=-1)
        _rate_stim = _spike_count_stim / STIMDUR
        _rate_stim_mean = np.mean(_rate_stim, axis=-1)

        out = _rate_stim_mean

        return out

    def get_thresholds(self, alpha=1, unresponsive_alpha=0, use_session_vars=True):
        if use_session_vars:
            session_values_center_responsive = (
                (self.spontaneous + alpha * self.stdspontaneous) / BLKDUR
            )[self.neuron_locations["center"]]

            session_values_off_center_responsive = (
                (self.spontaneous + alpha * self.stdspontaneous) / BLKDUR
            )[self.neuron_locations["off_center"]]

            if unresponsive_alpha == "max":
                session_values_off_center_unresponsive = (
                    max(
                        (self.spontaneous + unresponsive_alpha * self.stdspontaneous),
                        0.1,
                    )
                    / BLKDUR
                )[self.neuron_locations["off_center"]]
            else:
                session_values_off_center_unresponsive = (
                    (self.spontaneous + unresponsive_alpha * self.stdspontaneous)
                    / BLKDUR
                )[self.neuron_locations["off_center"]]

            self.thresholds = {
                "center_responsive": session_values_center_responsive,
                "off_center_responsive": session_values_off_center_responsive,
                "off_center_unresponsive": session_values_off_center_unresponsive,
            }
        else:
            locations = ["center", "off_center", "off_center"]
            conditions = ["_responsive", "_responsive", "_unresponsive"]
            alphas = [alpha, alpha, unresponsive_alpha]
            self.thresholds = {
                i + j: self._get_threshold(alpha=k, neuron_location=i)
                for i, j, k in zip(locations, conditions, alphas)
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
        n_ms = self.resp_train_blk.shape[-1]

        trains_blk = self.resp_train_blk[:, :, :, n_ms - BLKDUR : n_ms + 1]
        spike_count_blk = np.sum(trains_blk, axis=-1)
        rate_blk = spike_count_blk / BLKDUR

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
