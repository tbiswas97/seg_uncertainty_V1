from import_utils import loadmat_h5, _load
import import_utils
import toolbox as tb
import numpy as np
import os
from itertools import combinations
import pandas as pd
from scipy.spatial import KDTree
from collections import Counter
from vseg.src.vseg import SegmentationMap as PseudoSegmentationMap
from numpy.random import choice as choose
import torch

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
        #user defined attributes should go below line 46
        self.__dict__ = temp["Session"]
        self.df = None
        self.neuron_df = None
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
        """
        Outputs a dataframe with positional and index information for all center and off_center neurons  

        Parameters:
        ----------------
        self : Session object

        Returns:
        ----------------
        DataFrame with columns:
            DataFrame.index : int 
                a standard index for all rows 
            index : int
                The index of the neuron in the Session.df dataframe
            neuron : int 
                The index of the neuron in the experiment structure 
            xy_coord : list, float
                A 2-element list of coordinates with the origin defined in the center of the image
            pos : str
                states whether the neuron is considered "center" or "off_center"
            x : float 
                The first element of xy_coord
            y : float 
                The second element of xy_coord
            x_np : int 
                The first element of xy_coord with the origin defined in the upper left corner of the image
                (numpy array standard)
            y_np : int 
                The second element of xy_coord with the origin defined in the upper left corner of the image
                (numpy array standard)
            x_mpl : int 
                The first element of xy_coord with the origin defined in the lower left corner of the image
                (matplotlib plot standard)
            y_mpl : int 
                The second element of xy_coord with the origin defined in the lower left corner of the image
                (matplotlib plot standard)
        """
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
        squeeze = neurons.drop_duplicates(["neuron"]).reset_index()

        coords_df = pd.DataFrame(squeeze.xy_coord.to_list(),columns=["x","y"])

        tx_np = tb.slicer(lambda x:tb.transform_coord_system((x,0))[1])
        ty_np = tb.slicer(lambda y:tb.transform_coord_system((0,y))[0])

        tx_mpl = tb.slicer(lambda x:tb.transform_coord_system_mpl((x,0))[0])
        ty_mpl = tb.slicer(lambda y:tb.transform_coord_system_mpl((0,y))[1])

        
        out = pd.concat([squeeze,coords_df],axis=1)

        out["x_np"] = tx_np(out["x"])
        out["y_np"] = ty_np(out["y"])

        out["x_mpl"] = tx_mpl(out["x"])
        out["y_mpl"] = ty_mpl(out["y"])

        self.neuron_df = out

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

    #PSEUDONEURON FUNCTIONS FOR PSEUDOSEGMENTATION

    def _get_neuron_pos_KDTree(self,location="center",coord_type="mpl"):
        """"
        Returns a KDTree containing the coordinates of specified neurons:

        Parameters:
        ------------
        location : str
            Valid values are "center", "off_center", None
            if None the coordinates of ALL neurons are used 
        coord_type : str 
            "mpl", "np" or None 
            if None the coordinates are with origin as center
    
        Retruns:
        --------------------
        self.tree : KDTree
            see scipy.spatial.KDTree
        
        """
        xy_strs = ["x", "y"]

        if coord_type is not None:
            xy_strs = [s + "_" + coord_type for s in xy_strs]

        if self.neuron_df is not None: 
            ndf = self.neuron_df
        else:
            ndf = self.get_neuron_info()
        
        if location is not None:
            coords = ndf.loc[ndf.pos==location,[xy_strs[0],xy_strs[1]]].to_numpy()
            
            ndf_tree = ndf.loc[ndf.pos==location]
            ndf_tree.insert(1,"mapping_index",list(range(len(ndf_tree))))

            self.neuron_df = ndf_tree
        else:
            coords = ndf.loc[:,["x_mpl","y_mpl"]].to_numpy()

        tree = KDTree(coords)

        self.KDTree = tree

        return tree
    
    def _get_pseudo_grid(self,center_window=None,imsize=256,gridsize=15):
        """
        Returns the coordinates of a grid of pseudo-neurons across the center of the image
        (defined by center_window)

        Parameters:
        ----------
        center_window : int, default None
            since pseudoneurons can only be defined in the center, the length of the center window
        imsize : int, default 256
            the total size of the image to pseudosegment
        gridsize : int, default 15
            the resolution of the pseudogrid will be gridsize x gridsize

        Returns: 
        ---------
        coords : array of shape (gridsize x gridsize,2)
        """
        if center_window is not None:
            center_window = center_window
        else: 
            center_window = self.neuron_exclusion_parameters["thresh"]
        
        origin = imsize//2
        s = origin-center_window//2
        S = origin+center_window//2
        x = np.linspace(s,S,gridsize,dtype="int")
        grid_coords = np.asarray(np.meshgrid(x,x)).reshape((2,-1)).T  
        
        self.pseudogrid_coords = grid_coords 
        self.pseudogrid_size = gridsize

        return grid_coords

    def _get_pseudoneuron_sampling_weights(self,param,method="relu"):
        """
        Returns a sampling weight per neuron as a function of distance

        Parameters:
        ----------
        method : str, default "relu"
            Options are: 
                -relu : 
                -gaussian2d
                -softmax
        param : float 
            Has different meanings depending on the method used: 
                -relu : a distance threshold, neurons past this threshold have sampling weights of 0
                    other neurons have sampling weights that are scaled linearly according to distance
                -softmax : temperature 
                -gaussian2d : covariance 
            
        Returns: 
        ----------
        weights : array, float 
            shows sampling weights from [0,1], has shape (gridsize x gridsize, center_neurons)
        ncd : array, float 
            shows (max(distance) - distance) for each neuron, has shape (gridsize x gridsize, center_neurons) 
        distance : array, float 
            shows distance for each neuron, has shape (gridsize x gridsize, center_neurons) 
        nn_info[1] : array, int 
            shows index of neurons in ascending order of distance, has shape (center_neurons)

            
        """
        if method=="relu":
            tree = self.KDTree        
            grid = self.pseudogrid_coords
            #nearest neighbor information
            nn_info = tree.query(grid,k=len(tree.data))
            distances = nn_info[0]
            
            close_distances = distances.copy()
            close_distances[close_distances>param] = np.nan
            
            norm_close_distances = np.nanmax(close_distances,axis=1)[...,np.newaxis] - close_distances
            ncd = norm_close_distances
            weights = ncd.copy()
            
            sncd = np.nansum(ncd,axis=1)
            zero_vector_idxs = (sncd==0)
            zvs = zero_vector_idxs
            
            weights[~zvs] = (ncd/sncd[...,np.newaxis])[~zvs]
            
            
            if len(weights[zvs]) == 0:
                pass
            else:
                weights[zvs] = np.apply_along_axis(tb.nan_softmax,1,ncd[zvs])
            
            w = weights
            
            w[~(w==w)] = 0 
            
            assert w.shape == distances.shape
            
        self.pseudoneuron_sampling_weights = weights

        return weights, ncd, distances, nn_info[1]
    
    def _get_pseudoneuron_sampling_grid(self,weights):
        """
        Generates an instance of the pseudoneuron grid 

        Parameters:
        -------------
        weights : array, float 
            from self._get_pseudoneuron_sampling_weights
        
        Returns: 
        ---------
        pseudoneurons_grid : array, int 
            an array of shape (gridsize,gridsize) in which each element is assigned a neuron
        """
        #generate a LUT to map from the KDtree to actual neuron idxs

        weights = weights 
        
        pseudoneurons = np.asarray([
            choose(np.asarray(list(range(len(self.KDTree.data))),dtype="int"),
                p=weights[idx])
            for idx in range(weights.shape[0])
            ]
        )

        pseudoneurons_grid = pseudoneurons.reshape((self.pseudogrid_size,self.pseudogrid_size))

        self.pseudoneurons_grid = pseudoneurons_grid

        return pseudoneurons_grid

    def _lookup_delta_rsc(self,image_idx,neuron1,neuron2):
        if self.df is not None: 
            if 'delta_rsc' in self.df.columns:
                df = self.df
            else:
                self.df['delta_rsc'] = self.df['rsc_small'] - self.df['rsc_large']
        else: 
            self.df = self.get_df()
            self.df['delta_rsc'] = self.df['rsc_small'] - self.df['rsc_large']
            df = self.df

        df = df.loc[:,["delta_rsc","neuron1","neuron2","img_idx"]]

        delta_rsc = df.loc[
            (df.neuron1==neuron1)&
            (df.neuron2==neuron2)&
            (df.img_idx==image_idx),
            ["delta_rsc"]
        ]

        if len(delta_rsc) == 0:
            return np.nan
        else:
            return delta_rsc.values.squeeze()

    def _get_pseudoneuron_responses(
            self,
            img_idx,
            center_window=None,
            im_size=256,
            gridsize=15,
            activation_param=15,
            activation_method="relu",
            pseudoneurons_list=None):
        #use the PseudoSegmentationMap class (check imports) to generate a valid list of pairs
        if pseudoneurons_list is not None:
            pseudoneurons_list = pseudoneurons_list
        else:
            self.get_neuron_info()
            self._get_neuron_pos_KDTree()
            self._get_pseudo_grid(
                center_window=center_window,
                imsize=im_size,
                gridsize=gridsize
            )

            self._get_pseudoneuron_sampling_weights(param=activation_param, method=activation_method)

            self._get_pseudoneuron_sampling_grid(self.pseudoneuron_sampling_weights)

            pseudoneurons_list = np.ravel(self.pseudoneurons_grid)
            
        seg_map = PseudoSegmentationMap(2,self.pseudogrid_size,device="cpu")
        self._PseudoSegmentationMap_indices_object = seg_map
        seg_map_pairs = [tuple(pair) for pair in seg_map.px_pairs.tolist()]

        neuron_lut = {k:v for k,v in zip(list(self.neuron_df.mapping_index),list(self.neuron_df.neuron))}

        delta_rscs = np.asarray(
            [
                self._lookup_delta_rsc(
                    img_idx,
                    neuron_lut[
                        pseudoneurons_list[pair[0]]
                    ],
                    neuron_lut[
                        pseudoneurons_list[pair[1]]
                    ])
                        for pair in seg_map_pairs        
            ]
        )

        R = np.zeros(delta_rscs.shape)
        R[delta_rscs>0] = 1

        return R

    def get_pseudoneuron_response_table(
        self,
        n_trials,
        img_idx,
        center_window=None,
        im_size=256,
        gridsize=15,
        activation_param=15,
        activation_method="relu",
        pseudoneurons_list=None
    ):
        labels = ["responses_{}".format(i) for i in range(n_trials)]
        responses = []
        for i in range(n_trials):
            responses.append(self._get_pseudoneuron_responses)
        response_dict = {
            k:v for k,v in zip(labels,responses)
        }

        df = pd.DataFrame.from_dict(response_dict)

        self.responses_df = pd.concat([self.responses_df,df],axis=1)

    #def pseudosegment(
            #self,
            #image_idx,
            #k,
            #center_window=None,
            #im_size=None,
            #gridsize=15,
            #activation_param=15,
            #activation_method="relu",
            #lapreg = 5,
            #lr = 1e1,
            #max_iter = 50000,
            #tol = 1e-6,
            #sav_iter=True
    #):
        #"""
        #Performs a pseudosegmentation using the visual segmentation protocol: 

        #Parameters:
        #-----------
        #image_idx : int
            #From the session, the index of the image to segment 
        #center_window : optional, default None
            #if None, uses the neuron exclusion threshold radius
        #im_size : optional default, 256
            #if None, uses 256
        #gridsize : int
            #the gridsize to use for pseudosegmentation 
        #activation_param : optional
            #same as param in self._get_pseudoneuron_sampling_weights
        #activation_method : option
            #same as method in self._get_pseudoneuron_sampling_weights

        #Returns:
        #---------

        #"""
        #pass




